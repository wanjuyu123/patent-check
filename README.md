# **IPC/CPC Automatic Inference and Validation Tool (IPC/CPC 自动推断与验证工具)**

This project provides a robust, end-to-end solution for automatically inferring IPC/CPC classification codes for patents using Large Language Models (LLMs) like GPT-4o. It then validates the AI-generated codes against a local, expert-defined mapping to evaluate the model's accuracy.

本项目提供了一个端到端的自动化解决方案，利用大型语言模型（LLM，如 GPT-4o）为专利自动推断 IPC/CPC 分类号，并对照本地的专家知识库对推断结果进行验证和评估。

## **Features (主要特性)**

* **LLM-based Inference**: Leverages the power of models like GPT-4o to analyze patent titles and abstracts for accurate code inference.  
* **Local Validation**: Compares LLM results against a local "ground truth" mapping to quantitatively measure performance.  
* **Resumable Execution**: Uses checkpoints to save progress, allowing you to resume interrupted jobs without starting over.  
* **API Call Caching**: Caches every API request payload and its response, avoiding redundant calls and saving time and money on subsequent runs.  
* **Auto-Batching**: Dynamically adjusts the number of patents sent per API call to maximize throughput while respecting the model's context window limits.  
* **Highly Configurable**: Almost all parameters, including file paths, model names, and validation thresholds, can be configured via command-line arguments.

## **Getting Started (快速开始)**

### **Prerequisites (环境要求)**

* Python 3.8+  
* Git

### **Installation (安装步骤)**

1. **Clone the repository (克隆代码库):**  
   git clone \<your-repository-url\>  
   cd \<your-repository-name\>

2. **Create and activate a virtual environment (创建并激活虚拟环境):**  
   \# For macOS/Linux  
   python3 \-m venv venv  
   source venv/bin/activate

   \# For Windows  
   python \-m venv venv  
   .\\venv\\Scripts\\activate

3. **Install dependencies (安装依赖)**

### **Project Structure (项目结构)**

Your project should be organized with the following directory structure:

project-root/  
├── .gitignore  
├── mappings/  
│   └── Sensing.xlsx              \# Example mapping file for a domain  
├── patents/  
│   └── Sensing/  
│       └── Radar\_Technology.csv    \# Example patent data for a technology  
├── outputs/                        \# Generated output will be stored here (auto-created)  
├── .env  
├── .env.example  
├── run.py  
└── README.md

### **Configuration (配置)**

The script requires an OpenAI API key to function.

1. Create the environment file:  
   Copy the example file .env.example to a new file named .env. This file is listed in .gitignore and will not be committed to the repository.  
   cp .env.example .env

2. Add your API Key:  
   Open the .env file and add your OpenAI API key:  
   OPENAI\_API\_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

## **Usage (如何运行)**

You can run the script from the command line.

Basic execution (使用默认参数运行):  
This command will use the default settings specified in run.py (e.g., directories, model).  
python run.py

Execution with custom arguments (使用自定义参数运行):  
You can override the default settings using command-line flags.  
python run.py \--model gpt-4o \--auto-batch \--patents-dir /path/to/your/patents

### **Key Command-Line Arguments (主要命令行参数)**

| Argument | Description | Default Value |
| :---- | :---- | :---- |
| \--patents-dir | Directory containing the patent data, organized in domain subfolders. | patents |
| \--mappings-dir | Directory containing the expert mapping files (Excel/CSV). | mappings |
| \--outputs-dir | Directory where all results, caches, and state files will be saved. | outputs |
| \--model | The name of the OpenAI model to use for inference. | gpt-4o |
| \--auto-batch | If enabled, automatically calculates the optimal batch size. | False (disabled) |
| \--batch-size | Manually set the number of patents per API call (ignored if \--auto-batch). | 30 |
| \--tau-strong | The minimum A+B match rate to be considered "supported". | 0.35 |
| \--tau-min | The minimum A+B match rate to be considered "partially\_supported". | 0.15 |

## **Output (结果输出)**

The script generates all output in the outputs/ directory:

* outputs/raw/: Stores the raw JSON responses from every API call and cache files.  
* outputs/summary/: Contains a detailed CSV report for each individual technology subclass.  
* outputs/state/: Holds the checkpoints.json file for resuming execution.  
* outputs/summary\_all.csv: A final, consolidated report summarizing the results for all processed technologies.

## **License (许可证)**

This project is licensed under the MIT License. See the LICENSE file for details.
