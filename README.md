# PC Emporium RAG Tutorial

This repo contains the sample data for Contentful's RAG tutorial.

## What's in this repo

The `data/` folder contains three types of sample data for PC Emporium, a fictional PC components retailer:

- `data/faqs.json` — product FAQs about compatibility, setup, and components
- `data/policies/` — company policy documents covering returns, warranties, shipping, and more
- `data/blog-posts/` — longer-form guides on topics like choosing a GPU and building a PC

## Getting started

Follow the tutorial to build the RAG pipeline yourself. If you get stuck or want to compare your code against a working version, check the `completed` branch.

## Setup

1. Copy `.env.example` to `.env` and add your OpenAI API key
2. Create and activate a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate`
3. Install dependencies:
   `pip install -r requirements.txt`

## Acknowledgments

This project began as a learning exercise based on the RAG tutorial and
example repository by Sarah Barber / Contentful:

https://github.com/sarahbarberuk/rag-tutorial
https://www.contentful.com/blog/retrieval-augmented-generation-tutorial/

It has since been extended with persistent ChromaDB storage, metadata-based
document filtering, prompt refinement, retrieval testing, hallucination
controls, and source attribution.
