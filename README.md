# GPT-4 for Reliability Engineering Exam Assistant

## Overview
This project explores the capabilities of GPT-4 in the context of reliability engineering examinations. It utilizes the gpt-4o-mini model to assist in solving reliability engineering problems and demonstrate the potential of AI in technical education.

## Authors
- Clovis Piedallu (CentraleSupélec)
- Jérémy Mathet (CentraleSupélec)

## Project Purpose
The main objective is to leverage GPT-4-turbo's capabilities to:
- Analyze and solve reliability engineering problems
- Demonstrate AI's potential in technical assessments
- Explore the boundaries of AI assistance in engineering education

## Project Components

### RAG System (rag.py)
The Retrieval Augmented Generation (RAG) system is implemented in `rag.py`. It's based on the full documentation of [reliability python library](https://reliability.readthedocs.io/en/latest/), stored in a 20K lines markdown file. It provides:
- A comprehensive knowledge base built from reliability engineering documentation
- Integration with the reliability library's API documentation
- Intelligent chunking of documentation with hierarchy preservation
- Semantic search capabilities using embeddings
- Context-aware retrieval for accurate question answering

Key features:
- Uses text-embedding-3-small model for document embedding
- Implements hierarchical chunking for better context preservation
- Maintains section structure from the original documentation
- Efficient embedding storage and retrieval system
- Configurable chunk sizes and overlap

Full RAG and embeddings can be found in `rags` folder.

### Python Execution Agent (agents.py)
The Python execution environment, defined in `agents.py`, provides:
- Secure sandbox for executing Python code
- Virtual environment isolation
- Timeout protection
- Error handling and output capture
- Tool integration with GPT-4 function calling

Features:
- Safe execution of generated Python code
- Configurable timeout settings
- Cross-platform compatibility (Windows/Linux/Mac)
- Structured output handling

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- OpenAI API access

### Installation
1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
1. Create a `.env` file in the project root
2. Add your OpenAI API credentials:
```
OPENAI_API_KEY=your_api_key_here
```

## Evaluation and Results
This project was evaluated through a [Kaggle competition](https://www.kaggle.com/competitions/generative-ai-for-reliability-engineering/overview). The submission script is located in `kaggle_submission.py`, which generates predictions using our RAG and agent system.

### Performance
- Achieved accuracy: 0.82
- Final ranking: 3rd place on the private leaderboard
- Demonstrates strong performance in applying AI to reliability engineering problems

## Note
This project is part of the Gen AI for Risk and Reliability course at CentraleSupélec.
