"""
LLM Providers
=============

Unified interface for multiple LLM backends.
Supports: Ollama (local), Groq (free cloud), OpenAI, Anthropic
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLM(ABC):
    """Base LLM interface."""
    
    name: str = "base"
    supports_tools: bool = False
    supports_tool_choice: bool = False
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict:
        pass
    
    @abstractmethod
    def available(self) -> bool:
        pass


def _safe_json_loads(value: Any) -> Dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


class Ollama(LLM):
    """Local Ollama with tool support."""
    
    supports_tools: bool = True
    supports_tool_choice: bool = False

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.name = f"Ollama/{model}"
    
    def available(self) -> bool:
        try:
            import ollama
            ollama.list()
            return True
        except:
            return False
    
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict:
        import ollama
        kwargs = {"model": self.model, "messages": messages, "options": {"temperature": 0.7}}
        if tools:
            kwargs["tools"] = tools
        r = ollama.chat(**kwargs)
        msg = r["message"]
        result = {"content": msg.get("content", "")}
        if "tool_calls" in msg and msg["tool_calls"]:
            result["tool_calls"] = [
                {"id": f"call_{i}", "name": tc["function"]["name"], "arguments": tc["function"].get("arguments", {})}
                for i, tc in enumerate(msg["tool_calls"])
            ]
        return result


class Groq(LLM):
    """Groq cloud (FREE)."""
    
    supports_tools: bool = True
    supports_tool_choice: bool = True

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.key = os.environ.get("GROQ_API_KEY")
        self.model = model
        self.name = f"Groq/{model}"
    
    def available(self) -> bool:
        if not self.key:
            return False
        try:
            from groq import Groq
            return True
        except:
            return False
    
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict:
        from groq import Groq as G
        client = G(api_key=self.key)
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        r = client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        result = {"content": msg.content}
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            result["tool_calls"] = [
                {"id": t.id, "name": t.function.name, "arguments": _safe_json_loads(t.function.arguments)}
                for t in msg.tool_calls
            ]
        return result


class OpenAI(LLM):
    """OpenAI cloud."""
    
    supports_tools: bool = True
    supports_tool_choice: bool = True

    def __init__(self, model: str = "gpt-4o-mini"):
        self.key = os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.name = f"OpenAI/{model}"
    
    def available(self) -> bool:
        if not self.key:
            return False
        try:
            from openai import OpenAI  # type: ignore[reportMissingImports]
            return True
        except:
            return False
    
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict:
        from openai import OpenAI as O  # type: ignore[reportMissingImports]
        client = O(api_key=self.key)
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        r = client.chat.completions.create(**kwargs)
        msg = r.choices[0].message
        result = {"content": msg.content}
        if msg.tool_calls:
            result["tool_calls"] = [
                {"id": t.id, "name": t.function.name, "arguments": _safe_json_loads(t.function.arguments)}
                for t in msg.tool_calls
            ]
        return result


class Anthropic(LLM):
    """Anthropic Claude."""
    
    supports_tools: bool = True
    supports_tool_choice: bool = False

    def __init__(self, model: str = "claude-3-haiku-20240307"):
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.name = f"Claude/{model}"
    
    def available(self) -> bool:
        if not self.key:
            return False
        try:
            import anthropic  # type: ignore[reportMissingImports]
            return True
        except:
            return False
    
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict:
        import anthropic  # type: ignore[reportMissingImports]
        client = anthropic.Anthropic(api_key=self.key)
        
        system = ""
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        
        kwargs: Dict[str, Any] = {"model": self.model, "max_tokens": 4096, "messages": chat_msgs}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
                for t in tools
            ]
        
        r = client.messages.create(**kwargs)
        result: Dict[str, Any] = {"content": ""}
        tool_calls = []
        for block in r.content:
            if block.type == "text":
                result["content"] = block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result


def get_llm(name: Optional[str] = None) -> Optional[LLM]:
    """Get LLM by name or auto-detect."""
    providers = {"ollama": Ollama, "groq": Groq, "openai": OpenAI, "anthropic": Anthropic}
    
    if name and name.lower() in providers:
        llm = providers[name.lower()]()
        return llm if llm.available() else None
    
    # Auto-detect: Groq > Ollama > OpenAI > Anthropic
    for cls in [Groq, Ollama, OpenAI, Anthropic]:
        llm = cls()
        if llm.available():
            return llm
    return None
