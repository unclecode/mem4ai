<!-- <p align="center">
  <img src="docs/images/memtor-logo.png" width="400px" alt="Memtor - Intelligent Memory Layer for AI and LLM Applications">
  <p align="center">
    <a href="https://memtor.readthedocs.io">Documentation</a>
    ·
    <a href="https://github.com/unclecode/memtor/issues">Report Bug</a>
    ·
    <a href="https://github.com/unclecode/memtor/issues">Request Feature</a>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/memtor/">
    <img src="https://badge.fury.io/py/memtor.svg" alt="PyPI version">
  </a>
  <a href="https://pepy.tech/project/memtor">
    <img src="https://pepy.tech/badge/memtor/month" alt="Downloads">
  </a>
  <a href="https://github.com/unclecode/memtor/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/unclecode/memtor.svg" alt="License">
  </a>
  <a href="https://github.com/unclecode/memtor/stargazers">
    <img src="https://img.shields.io/github/stars/unclecode/memtor.svg" alt="Stars">
  </a>
</p> -->

# 🧠 Memtor: Intelligent Memory Layer for AI and LLM Applications

Memtor enhances AI assistants and agents with an intelligent memory layer, enabling personalized and context-aware AI interactions. Perfect for building smarter chatbots, AI assistants, and autonomous systems that remember and learn.

## ✨ Core Features

- 🔀 **Multi-Level Memory**: User, Session, and AI Agent memory retention
- 🎯 **Adaptive Personalization**: Continuous improvement based on interactions
- 🚀 **Fast Local Storage**: Efficient on-disk storage with quick access
- 🔍 **Advanced Semantic Search**: Find relevant memories quickly
- 🏷️ **Flexible Metadata**: Tag and filter memories easily
- 🔧 **Customizable Strategies**: Adapt embedding and storage to your needs
- 💾 **Persistent Storage**: Memories that last across sessions

## 🚀 Quick Start

Install Memtor:

```bash
pip install memtor
```

Here's a simple example of how to use Memtor:

```python
from memtor import Memtor

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
- 📝 Note-taking applications with semantic search
- 🎮 Adaptive gaming experiences
- 🏥 Patient history management in healthcare

## 📚 Documentation

For full documentation, visit [memtor.readthedocs.io](https://memtor.readthedocs.io).

## 🤝 Contributing

We welcome contributions! Check out our [contribution guidelines](CONTRIBUTING.md) for more information.

## 📄 License

Memtor is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by Unclecode. (https://x.com/unclecode)

Give Memtor a star ⭐ if you find it helpful!