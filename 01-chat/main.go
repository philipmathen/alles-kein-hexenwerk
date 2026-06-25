// Schritt 1: Reiner Chat mit dem LLM – noch keine Tools.
//
// In diesem Schritt sieht man das absolute Minimum eines Agenten:
// eine Schleife, die User-Eingaben einsammelt, an das Modell schickt
// und die Antwort ausgibt. Die Konversation wird als Liste von Nachrichten
// mitgeführt, damit das Modell den Verlauf kennt.
package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/lpernett/godotenv"
)

func main() {
	loadEnv()

	client := anthropic.NewClient()

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(&client, getUserMessage)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// loadEnv sucht ab dem Arbeitsverzeichnis aufwärts nach einer .env Datei.
// Dadurch funktioniert jeder Schritt – egal ob aus dem Repo-Root
// (go run ./01-chat) oder direkt aus dem Ordner (go run .) gestartet.
func loadEnv() {
	dir, err := os.Getwd()
	if err != nil {
		log.Fatalf("could not determine working directory: %v", err)
	}
	for {
		envPath := filepath.Join(dir, ".env")
		if _, err := os.Stat(envPath); err == nil {
			if err := godotenv.Load(envPath); err != nil {
				log.Fatalf("Error loading .env file: %v", err)
			}
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	log.Fatal(".env file not found")
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with claude (ctrl + c to quit)")

	for {
		// Input vom User entgegennehmen
		fmt.Print("\x1b[38;5;39mUser\x1b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		// Antwort vom Modell generieren und an die Konversation anhängen
		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}
		conversation = append(conversation, message.ToParam())

		// Antwort vom Modell anzeigen
		for _, content := range message.Content {
			switch content.Type {
			case "text":
				fmt.Printf("\x1b[38;5;208mClaude\x1b[0m: %s\n", content.Text)
			}
		}
	}
	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (anthropic.Message, error) {
	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaudeHaiku4_5,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})
	return *message, err
}
