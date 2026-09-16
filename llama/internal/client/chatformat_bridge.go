package client

import (
	"github.com/coditary/wuji-core/chatformat"
	"github.com/coditary/wuji-core/pkg/driver"
)

func toChatformatMessages(msgs []driver.ChatMessage) []chatformat.Message {
	out := make([]chatformat.Message, 0, len(msgs))
	for _, m := range msgs {
		item := chatformat.Message{
			Role:       chatformat.Role(m.Role),
			Content:    m.Content,
			Name:       m.Name,
			ToolCallID: m.ToolCallID,
		}
		for _, tc := range m.ToolCalls {
			item.ToolCalls = append(item.ToolCalls, chatformat.ToolCall{
				ID: tc.ID, Name: tc.Name, Args: tc.Args,
			})
		}
		out = append(out, item)
	}
	return out
}

func toChatformatTools(tools []driver.ChatToolDef) []chatformat.ToolDef {
	out := make([]chatformat.ToolDef, 0, len(tools))
	for _, t := range tools {
		out = append(out, chatformat.ToolDef{
			Name: t.Name, Description: t.Description, Parameters: t.Parameters,
		})
	}
	return out
}
