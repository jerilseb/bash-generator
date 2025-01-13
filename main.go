package main

/*
#cgo linux LDFLAGS: -ljack -lasound

#include <jack/jack.h>
#include <alsa/asoundlib.h>

// JACK error callback: does nothing
static void jackErrorCallback(const char *msg) {}

// ALSA error callback: does nothing
static void alsaErrorCallback(const char *file, int line, const char *function, int err, const char *fmt, ...) {}

// Attempt to set JACK error handler
static void setJackErrorHandler() {
    // If libjack is present, this will override the default error handler.
    jack_set_error_function(jackErrorCallback);
}

// Attempt to set ALSA error handler
static void setAlsaErrorHandler() {
    // If libasound is present, this will override the default error handler.
    snd_lib_error_set_handler(alsaErrorCallback);
}
*/
import "C"

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/briandowns/spinner"
	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"github.com/gordonklaus/portaudio"
	"github.com/joho/godotenv"
)

// openAIChatRequest is the JSON structure we send to the Chat Completion endpoint.
type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []map[string]string `json:"messages"`
	Temperature float64             `json:"temperature"`
}

// openAIChatResponse is a partial structure for the response from the Chat Completion endpoint.
type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// openAITranscriptionResponse is a partial structure for the Whisper transcription response.
type openAITranscriptionResponse struct {
	Text string `json:"text"`
}

func init() {
	// Replicate JACK_NO_START_SERVER=1
	os.Setenv("JACK_NO_START_SERVER", "1")

	// Attempt to set JACK and ALSA error handlers via CGO.
	// If the libraries are not installed or linking fails, you might get build/link errors.
	// The Python code silently passes on OSError/AttributeError;
	// in Go/CGO, linking issues can appear at build time instead of runtime.
	C.setJackErrorHandler()
	C.setAlsaErrorHandler()
}

func main() {
	if err := run(); err != nil {
		fmt.Printf("An error occurred: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	_ = godotenv.Load()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file")
	}

	if err := portaudio.Initialize(); err != nil {
		return fmt.Errorf("failed to initialize portaudio: %w", err)
	}
	defer portaudio.Terminate()

	const (
		channels       = 1
		sampleRate     = 44100
		framesPerChunk = 1024
	)

	in := make([]int16, framesPerChunk)

	// Create an input stream
	stream, err := portaudio.OpenDefaultStream(channels, 0, float64(sampleRate), framesPerChunk, in)
	if err != nil {
		return fmt.Errorf("failed to open audio stream: %w", err)
	}
	defer stream.Close()

	// Start stream
	if err := stream.Start(); err != nil {
		return fmt.Errorf("failed to start audio stream: %w", err)
	}

	// We will store recorded data in a buffer
	var recordedData []int16

	// Use a spinner to replicate the Halo spinner from Python
	s := spinner.New(spinner.CharSets[11], 100*time.Millisecond)
	s.Suffix = " Recording"
	s.Start()
	defer s.Stop()

	// We will stop recording when the user hits Enter OR when Ctrl+C is pressed.
	var stopRecording int32

	// Goroutine to wait for Enter
	go func() {
		// Wait for user to press Enter
		bufio.NewReader(os.Stdin).ReadString('\n')
		atomic.StoreInt32(&stopRecording, 1)
	}()

	// Also handle Ctrl+C
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		atomic.StoreInt32(&stopRecording, 1)
	}()

	// Recording loop
	for atomic.LoadInt32(&stopRecording) == 0 {
		if err := stream.Read(); err != nil && err != io.EOF {
			s.Stop()
			return fmt.Errorf("error reading from audio stream: %w", err)
		}
		// Append the current chunk to our recorded buffer
		recordedData = append(recordedData, in...)
	}

	// Stop stream
	if err := stream.Stop(); err != nil {
		return fmt.Errorf("failed to stop audio stream: %w", err)
	}

	// Write to a temporary WAV file
	tempFileName := "temp.wav"
	if err := writeWavFile(tempFileName, recordedData, channels, sampleRate); err != nil {
		return fmt.Errorf("failed to write wav file: %w", err)
	}
	defer os.Remove(tempFileName) // Clean up after done

	// Transcription request
	s.Suffix = " Transcribing audio..."
	transcribedText, err := transcribeAudio(apiKey, tempFileName)
	if err != nil {
		s.Stop()
		return fmt.Errorf("error transcribing audio: %w", err)
	}

	// Send transcribed text to GPT-4 to get a Bash command
	s.Suffix = " Generating command..."
	generatedCommand, err := generateBashCommand(apiKey, transcribedText)
	if err != nil {
		s.Stop()
		return fmt.Errorf("error generating command: %w", err)
	}

	// Stop the spinner and print the result
	s.Stop()
	fmt.Printf("\n%s\n", strings.TrimSpace(generatedCommand))
	return nil
}

// writeWavFile writes the provided int16 samples into a WAV file with given channels and sampleRate.
func writeWavFile(filename string, samples []int16, numChans, sampleRate int) error {
	// Create the output file
	outFile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer outFile.Close()

	// Prepare WAV encoder
	enc := wav.NewEncoder(outFile, sampleRate, 16, numChans, 1)

	// Convert []int16 into []int for the audio library
	buf := &audio.IntBuffer{
		Format: &audio.Format{
			NumChannels: numChans,
			SampleRate:  sampleRate,
		},
		Data:           int16ToIntSlice(samples), // Convert here
		SourceBitDepth: 16,
	}

	if err := enc.Write(buf); err != nil {
		return err
	}
	if err := enc.Close(); err != nil {
		return err
	}
	return nil
}

// int16ToIntSlice converts a slice of int16 to a slice of int.
func int16ToIntSlice(in []int16) []int {
	out := make([]int, len(in))
	for i, v := range in {
		out[i] = int(v)
	}
	return out
}

// transcribeAudio sends the WAV file to OpenAI's Whisper endpoint and returns the transcribed text.
func transcribeAudio(apiKey, filePath string) (string, error) {
	// Prepare multipart form
	var b bytes.Buffer
	w := multipart.NewWriter(&b)

	// Add file
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	fw, err := w.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(fw, file); err != nil {
		return "", err
	}

	// Add model field
	if err := w.WriteField("model", "whisper-1"); err != nil {
		return "", err
	}

	if err := w.Close(); err != nil {
		return "", err
	}

	// Prepare the request
	req, err := http.NewRequest("POST", "https://api.openai.com/v1/audio/transcriptions", &b)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", w.FormDataContentType())

	// Execute request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Check status
	if resp.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("non-200 status code: %d - %s", resp.StatusCode, string(responseBody))
	}

	// Parse JSON
	var transcription openAITranscriptionResponse
	if err := json.NewDecoder(resp.Body).Decode(&transcription); err != nil {
		return "", err
	}
	return transcription.Text, nil
}

// generateBashCommand sends the transcribed text to the GPT-4 Chat Completions
// endpoint, instructing it to return a single Bash command.
func generateBashCommand(apiKey, userText string) (string, error) {
	payload := openAIChatRequest{
		Model: "gpt-4",
		Messages: []map[string]string{
			{
				"role":    "system",
				"content": "You convert natural language instructions into a single valid Bash command. Print the command in plain text without any formatting",
			},
			{
				"role":    "user",
				"content": userText,
			},
		},
		Temperature: 0.0,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("non-200 status code: %d - %s", resp.StatusCode, string(responseBody))
	}

	var chatResp openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", err
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from chat completion")
	}
	return chatResp.Choices[0].Message.Content, nil
}
