<!-- <p align="center">
  <img src="docs/images/Mem4AI-logo.png" width="400px" alt="Mem4AI - Intelligent Memory Layer for AI and LLM Applications">
  <p align="center">
    <a href="https://Mem4AI.readthedocs.io">Documentation</a>
    ·
    <a href="https://github.com/unclecode/Mem4AI/issues">Report Bug</a>
    ·
    <a href="https://github.com/unclecode/Mem4AI/issues">Request Feature</a>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/Mem4AI/">
    <img src="https://badge.fury.io/py/Mem4AI.svg" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/project/Mem4AI">
    <img src="https://pepy.tech/badge/Mem4AI/month" alt="Downloads">
  </a>
  <a href="https://github.com/unclecode/Mem4AI/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/unclecode/Mem4AI.svg" alt="License">
  </a>
  <a href="https://github.com/unclecode/Mem4AI/stargazers">
    <img src="https://img.shields.io/github/stars/unclecode/Mem4AI.svg" alt="Stars">
  </a>
</p> -->

# 🧠 Mem4AI: A LLM Friendly memory management library.

Mem4AI enhances AI assistants and agents with an intelligent memory layer, enabling personalized and context-aware AI interactions. Perfect for building smarter chatbots, AI assistants, and autonomous systems that remember and learn.

## ✨ Core Features

- 🔀 **Multi-Level Memory**: User, Session, and AI Agent memory retention
- 🎯 **Adaptive Personalization**: Continuous improvement based on interactions
- 🚀 **Fast Local Storage**: Efficient on-disk storage with quick access
- 🔍 **Advanced Semantic Search**: Find relevant memories quickly
- 🏷️ **Flexible Metadata**: Tag and filter memories easily
- 🔧 **Customizable Strategies**: Adapt embedding and storage to your needs
- 💾 **Persistent Storage**: Memories that last across sessions

## 🚀 Quick Start

Install Mem4AI:

```bash
pip install Mem4AI
```

Here's a simple example of how to use Mem4AI:

```python
from mem4ai import Memtor

# Initialize Memtor
memtor = Memtor()

# Add a memory
memory_id = memtor.add_memory(
    "The user prefers dark mode in all applications",
    metadata={"preference": "ui", "mode": "dark"},
    user_id="alice"
)

# Search for memories
results = memtor.search_memories("user interface preferences", user_id="alice")
print(results[0].content)  # Output: The user prefers dark mode in all applications

# Update a memory
memtor.update_memory(
    memory_id,
    "The user prefers dark mode, except for document editing",
    metadata={"preference": "ui", "mode": "dark", "exception": "document_editing"}
)

# Delete a memory
memtor.delete_memory(memory_id)
```

## 🛠️ Use Cases

- 🤖 Chatbots with long-term memory
- 📊 Personalized recommendation systems
- 🧠 Knowledge management systems
- 🎯 Context-aware AI assistants

## 📚 Documentation

Coming soon!

## 🤝 Contributing

We welcome contributions! Check out our [contribution guidelines](CONTRIBUTING.md) for more information.

## 📄 License

Mem4AI is released under the Apache 2.0 license. See [LICENSE](LICENSE) for more information.

---

Built with ❤️ by Unclecode. (https://x.com/unclecode)

Give Mem4AI a star ⭐ if you find it helpful!