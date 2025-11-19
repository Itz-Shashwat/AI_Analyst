# LLM-Powered Local AI Data Analysis (100% Privacy-Preserving)

This project provides an **LLM-powered automated data-analysis system
that works entirely on your local machine**, ensuring **zero data
sharing**, **zero cloud dependency**, and complete **confidentiality**
of your datasets.\
Your data never leaves your system --- all analysis, code generation,
and execution happen locally.

## 🚀 Overview

We fine-tuned **Qwen 2.5 (3B)** using **LoRA PEFT**, resulting in a
lightweight but powerful model specialized for **Pandas data analysis
tasks**.\
The model converts natural-language questions into optimized Pandas
code, allowing analysts and developers to interact with data
conversationally without exposing any private files.

This system uses a **server--client design**: - **Server:** Hosts the
fine-tuned model and exposes a `/predict` API. - **Client:** A UI where
the user uploads a CSV locally and types a question.\
The file never leaves the device --- only the *question* is sent to the
server.

## 🔒 Key Features

-   **No data upload or sharing --- 100% local processing**
-   **Fine-tuned Qwen-2.5 3B** model for data-analysis tasks
-   **LoRA-based lightweight training (\~9M trainable params)**
-   **Understands natural-language questions and returns Pandas code**
-   **Works with CSVs, Excel, SQL extracts, etc.**
-   **REST API server for easy integration**
-   **Front-end client for smooth user interaction**

## 🧠 Model Training Details

### Base Model

-   **Qwen 2.5 -- 3B Instruct**

### Fine-Tuning Method

-   **LoRA (Low-Rank Adaptation) using PEFT**
-   \~**9 million trainable parameters**

### Training Dataset

The model was trained on a custom JSON dataset containing:

    {
      "instruction": "User question about a dataframe",
      "output": "Corresponding Pandas code"
    }

## ⚙️ Hyperparameters

  Hyperparameter          Value
  ----------------------- ----------
  Learning Rate           **2e-4**
  LoRA Rank (r)           16
  LoRA Alpha              32
  LoRA Dropout            0.05
  Batch Size              2--4
  Gradient Accumulation   4
  Epochs                  3--5
  Warmup Ratio            0.1
  Max Seq Length          2048

## 🌐 Server & Client Setup

### Backend Server

The server: - Loads the fine-tuned model and LoRA weights\
- Runs entirely on-device\
- Exposes a `/predict` endpoint\
- Returns model-generated Pandas code

### Front-End Client

The client: - Lets users **upload data locally**\
- Sends only the **text question** to the server\
- Executes returned code locally\
- Displays results as tables, charts, or summaries

## 🔐 Privacy & Security Guarantee

This system ensures: - Your dataset is **never uploaded** - No cloud
inference\
- No external API calls\
- All processing happens **locally**

Perfect for confidential settings: - Finance\
- Healthcare\
- Enterprise data\
- Compliance environments
