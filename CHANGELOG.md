# Changelog

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