# Changelog

## [2024-11-01 16:20:40] ## [2024-11-01 - 16:20] Version Update and New Strategy Inclusion
Updated the package version to 0.1.1. Included 'KnowledgeExtractionStrategy' in the package exports to enhance functionality.

## [31-Oct-2024] Enhanced Memory Storage and Retrieval System

### Added
- Timestamp tracking for all memories
- New indexed storage capabilities:
  - Timestamp-based indexing
  - Metadata-based indexing
  - Multi-dimensional filtering
- New retrieval methods:
  - `find_recent`: Get latest memories with optional filters
  - `find_by_time`: Query memories within time ranges
  - `find_by_meta`: Search using metadata combinations
- Separate LMDB environments for efficient index management

### Enhanced
- Memory class with automatic timestamp tracking
- Storage strategy with improved filtering capabilities
- LMDB storage implementation with index management
- Deletion handling with proper index cleanup

### Developer Notes
- Added comprehensive test suite for storage and retrieval
- Implemented efficient index updating mechanisms
- Added support for complex queries combining time and metadata filters
- Improved memory cleanup with proper index maintenance

## [31-Oct-2024] Knowledge Extraction for Enhanced Memory Context

### Added
- Knowledge extraction system for memories with configurable strategies
  - Support for LLM-based and simple extraction methods
  - Automatic context generation from user-assistant interactions
  - Configurable extraction settings in config manager
- New context field in Memory class to store extracted knowledge
- Extended memory creation to handle both user messages and assistant responses

### Changed
- Updated Memory class to support context storage and history
- Modified Memtor initialization to include extraction strategy configuration
- Enhanced memory update system to track context changes in history
- Expanded add_memory method to process combined user-assistant interactions

### Configuration
- Added new extraction configuration options:
  - `extraction.enabled`: Toggle knowledge extraction
  - `extraction.type`: Choose between 'llm' or 'simple' strategies
  - `extraction.model`: Specify LLM model for extraction
  - `extraction.store_full_response`: Control response storage
  - `extraction.extraction_timeout`: Set processing timeout
  - `extraction.retries`: Configure retry attempts

### Developer Notes
- Added comprehensive test suite for knowledge extraction functionality
- Implemented two test approaches:
  - Complex extraction with detailed context structure
  - Simple summary-based extraction with keywords