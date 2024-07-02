import os
import uuid
import psycopg2
import time
import re
import asyncio
import cohere
import numpy
import streamlit as st
import pdfkit
import json
import requests
import tempfile
import mistune
import markdown as md
import psycopg2
import html2text
from typing import List, Tuple, Dict
from pinecone import Pinecone, ServerlessSpec
import openai
import os
import pymupdf
import tiktoken
import google.generativeai as gemini
from PIL import Image
from PIL import PngImagePlugin  # important to avoid google_genai AttributeError
import json
import hashlib
from dotenv import load_dotenv
from classifier import Classifier
from tenacity import retry, stop_after_attempt, wait_random_exponential
import aiohttp

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API = os.getenv("COHERE_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

gemini.configure(api_key=GEMINI_API_KEY)

client = openai.OpenAI(base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY)

SysPromptDefault = "You are now in the role of an expert AI."
GenerationPrompt = """You are an expert AI whose task is to ANSWER the user's QUESTION using the provided CONTEXT.
Forget everything you know, Fully rely on CONTEXT to provide the answer.
Follow these steps:
1. Think deeply and multiple times about the user's QUESTION. You must understand the intent of their question and provide the most appropriate answer.
2. Choose the most relevant content from the CONTEXT that addresses the user's question and use it to generate an answer.
Formating Instructions:
Respond only in markdown format; don't use big headings"""
QuestionRouter = """ You are an expert investor, You must identify if the provided CONTEXT can answer the user QUESTION.  
1 vectorstore : The provided CONTEXT is sufficient to answer the question.
2 missing_information : The provided CONTEXT does not contains the answer.
output options: 'vectorstore' OR 'missing_information'.The output must be a valid JSON.Do not add any additional comments.
Output format:
{
    "datasource":"identified option"
}
Return the a valid JSON with a single key 'datasource' and no preamble or explanation. Question to identify: QUESTION """
MissingInformation = """You are an expert in identifying missing information in a given CONTEXT to answer a QUESTION effectively. Your task is to analyze the CONTEXT, and pinpoint the missing content needed for each QUESTION. Take your time to process the information thoroughly and provide a list output without any additional comments. The output format should be valid markdown list , without any additional comments: 
"""
SummaryPrompt = """You are an expert AI specializing in document summarization. You have been refining your skills in text summarization and data extraction for over a decade, ensuring the highest accuracy and clarity. Your task is to read raw data from a PDF file page by page and provide a detailed summary of the CONTEXT while ensuring all numerical data is included in the summary without alteration. The output should be in Markdown format, with appropriate headers and lists to enhance readability. Follow these instructions:
1.Summarize the Text: Provide a concise summary of the CONTEXT, capturing the main points and essential information.
2.Retain Numerical Data: Ensure all numerical data (e.g., dates, statistics, financial figures, percentages, measurements) is included in the summary.
3.Markdown Format: Format the output in Markdown, using headers, lists, and other Markdown elements appropriately.
Note: Whenever the CONTEXT is about a TEAM, DO NOT summarize; instead, output the content in a neat markdown format with Names and previous designation of the TEAM.
"""
IndustryPrompt = """You are a business strategy consultant. You have been identifying niche markets and industries for companies across various sectors for over 20 years. Your expertise lies in analyzing detailed CONTEXT to accurately pinpoint the niche and industry of a business.
Objective: Identify the niche and industry of a business by analyzing the provided CONTEXT.
Steps to follow:
Read the context: Carefully read the provided information to understand the business's products, services, target audience, and unique value propositions.
Determine the industry: Based on the provided CONTEXT, identify the primary industry to which the business belongs. Consider factors such as the type of products/services offered, the market served, and industry-specific terminology.
Identify the niche: Analyze the details to pinpoint the specific niche within the industry. Look for unique aspects of the business, specialized market segments, or specific customer needs that the business addresses.
Provide output in JSON format: Clearly state the identified industry and niche in a JSON format. Ensure your reasoning supports the identified industry and niche.The output should JSON ,Do not add any additional format.
Output format:
{
  "industry": "Identified industry here",
  "niche": "Identified niche here",
  "reasoning": "Explanation of how the industry and niche were identified based on the context"
}
Take a deep breath and work on this problem step-by-step.
"""

Investment = """You are a professional financial analyst with over 20 years of experience in evaluating sectors for investment potential. You specialize in providing comprehensive qualitative and quantitative analyses to assess the investment potential of various business sectors. Your expertise includes evaluating risk factors and potential returns to offer prudent investment advice.

### Objective:
Your primary goal is to deliver an accurate, in-depth evaluation of the business sector described in the provided CONTEXT. You will critically analyze the key sections and provide a grade for each section on a scale from 1 to 10 based on its investment potential. A higher grade indicates a higher potential return on investment (ROI) or a higher Sharpe ratio, while a lower grade indicates higher risk or lower expected returns. Approach each section with a conservative mindset to ensure a realistic and prudent assessment.

---

### Instructions:

### Step 1: Sector Identification and Key Sections Extraction

1. **Identify the Sector:**
   - Determine the specific sector you will analyze.
   - Clearly state the sector name in your final output.

2. **Extract Sections to be Graded:**
   - Extract and list the sections to be evaluated from the provided CONTEXT, ensuring only the KEYS specified in the CONTEXT are used.
   - Validate that the extracted sections strictly match the keys from the CONTEXT, without adding or altering any sections.

### Step 2: Grading Each Section

1. **Assign a Grade for Each Key in the Context:**
   - Assign a grade between 1 and 10 for each section.
   - Higher grades should correspond to higher investment potential, considering ROI or Sharpe ratio, while lower grades reflect higher risks or lower returns.
   -Consider conversative approach for the grading  because investment is a risk investors can loss money 

2. **Provide Detailed Reasoning:**
   - Ensure the grades given are backed by comprehensive and in-depth explanations.
   - Incorporate detailed reasoning from both qualitative and quantitative perspectives.
   - Support your assessments with numerical data and statistical analysis if present.
   - Ensure your evaluations are robust and well-supported with concrete evidence and thorough analysis.
   
3. **Assign a Weight to Each Section:**
   - Determine the importance of each section relative to the overall investment potential. Assign a weight between 0 and 1 for each section, ensuring the total weights add up to 1.
   - Higher weights should be given to sections that are more critical to the investment decision-making process.


### Step 3: Overall Sector Score Calculation

**Calculate Weighted Scores**:
   - **Weighted Score Calculation**: Multiply each score by its corresponding weight and sum these to get the total weighted score.
     \[
     \text{Total Score} = \sum (\text{Score}_i \times \text{Weight}_i)
     \]
### Output Format:

1. **JSON Structure:**
   - Ensure the output is in a valid JSON format with the following structure:

```json
{
  "sector": "Insert Sector Name Here",
  "sections": [
    {
      "section": "Insert Key from Context Here",
      "score": "Insert Grade Here (1-10)",
      "weight": "Insert Weight Here (0-1)",
      "reasoning": "Insert detailed qualitative and quantitative analysis here"
    },
    {
      "section": "Insert Key from Context Here",
      "score": "Insert Grade Here (1-10)",
      "weight": "Insert Weight Here (0-1)",
      "reasoning": "Insert detailed qualitative and quantitative analysis here"
    }
  ],
  "overall_score": "Insert overall score here"// replace with calculated score using the formula 
}
```

### Key Requirements:

1. **Accuracy:**
   - Ensure each section’s grade is backed by comprehensive reasoning. Cross-verify each grade with both qualitative and quantitative justifications.

2. **Contextual Relevance:**
   - Use only the keys provided in the CONTEXT. Avoid introducing new sections or altering the provided keys.

3. **Conservativeness:**
   - Adopt a critical and conservative approach in grading, reflecting realistic and prudent investment potential. Lower scores should be preferred to emphasize investment risks and the possibility of capital loss.

### Important Notes:
- The CONTEXT will be provided in the following format:
```json
{
  "key1": "value1",
  "key2": "value2",
  ...
}
```
- Only use the KEYS (e.g., key1, key2) to identify the sections to be graded. D0 NOT ADD or ALTER any sections beyond the KEYS present in the CONTEXT. Grade only the sections that match the KEYS provided in the CONTEXT.
-If the CONTEXT has one or no KEY return a JSON with the 'keys are already graded'.
Take a deep breath and work on this problem step-by-step.

---
"""

queries = [
    "What is the company's product/service, and what are its key features?",
    "Who is the target customer for the company's product/service, and what problem does it solve for them?",
    "What are the company's revenue streams?",
    "How does the company price its products/services?"
    "What are the key cost drivers and profit margins for the company?",
    "What opportunities for growth and expansion does the company foresee?",
    "Who is the target market for the company's product/service, and how does the company plan to reach them?",
    "What sales channels and distribution partnerships does the company have in place?",
    "How is the company's marketing budget allocated?",
    "What is the company's historical financial performance, including growth rate?",
    "What are the company's projected revenue, expenses, and profits for the future and cash flow projections?",
    "What is the founder's experience and track record, along with the key team members' bios, background checks, and their roles and responsibilities?",
    "How does the company's product/service differentiate itself from competitors in the market?",
    "What issue or challenge is the company addressing?",
]

document_processing_event = asyncio.Event()
document_processing_event.set()


def get_digest(pdf_content):
    h = hashlib.sha256()
    h.update(pdf_content)  # Hash the binary content of the PDF
    return h.hexdigest()


def response(
    message: object,
    model: object = "meta-llama/llama-3-70b-instruct:nitro",
    SysPrompt: object = SysPromptDefault,
    temperature: object = 0.2,
) -> object:
    """
    :rtype: object
    """
    client = openai.OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    messages = [
        {"role": "system", "content": SysPrompt},
        {"role": "user", "content": message},
    ]

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        print("RETRY")
        return client.chat.completions.create(**kwargs)

    try:
        response = completion_with_backoff(
            model=model,
            messages=messages,
            temperature=temperature,
            frequency_penalty=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")


def number_of_tokens(texts: List[str]) -> List[int]:
    """
    Calculate the number of tokens in a batch of strings.
    """
    model = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encodings = model.encode_batch(texts)
    num_of_tokens = [len(encoding) for encoding in encodings]
    return num_of_tokens


def limit_tokens(input_string, token_limit=5500):
    """

    Limit tokens sent to the model

    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return encoding.decode(encoding.encode(input_string)[:token_limit])


def extract_image_content(pixmap_list: List[pymupdf.Pixmap], text: str) -> List[str]:
    "Takes image path and extract information from it, and return it as text."

    # Start Classifier inference session
    classifier = Classifier("graph_classifierV2_B.onnx")
    # Model for img to text
    model = gemini.GenerativeModel("gemini-1.5-flash")

    description_prompt = f"You are provided with the images extracted from a pitch-deck and some text surrounding the image from the same pitch deck. Extract all the factual information that the image is trying to communicate through line charts, area line charts, bar charts, pie charts, tables exectra. Use OCR to extract numerical figures and include them in the information. If the image does not have any information like its a blank image or image of a person then response should be NOTHING. Do not add any additional comments or markdown, just give information. \n\n SURROUNDING TEXT \n\n{text}"

    img_list = []

    for pixmap in pixmap_list:
        try:
            img_list.append(
                Image.frombytes(
                    mode="RGB", size=(pixmap.width, pixmap.height), data=pixmap.samples
                )
            )
        except Exception as e:
            print(e)

    graph_image = classifier.classify(img_list)
    print(graph_image)

    response_list = []

    for idx, is_graph in enumerate(graph_image):
        if is_graph:
            response = model.generate_content(
                [description_prompt, img_list[idx]], stream=False
            )
            print("\n\n", response.text, "\n\n")
            response_list.append(str(response.text))

    return response_list


def extract_content(pdf_content: bytes) -> List[Tuple[str, int]]:
    """
    Takes PDF(bytes) and return a list of tuples containing text(including textual and image content)
    and page number containing that text.
    """
    print("Extract content called ")
    pdf_doc = pymupdf.open(stream=pdf_content, filetype="pdf")

    pages_content = []
    refered_xref = []
    for page_number in range(pdf_doc.page_count):
        page_content = ""

        # extracting text content
        page = pdf_doc.load_page(page_number)
        text_content = str(page.get_text()).replace("\n", "\t")
        page_content += text_content

        # extracting image content
        image_list = page.get_image_info(xrefs=True)
        pixmap_list = []
        for img_info in image_list:
            xref = img_info["xref"]
            if xref not in refered_xref:
                # if xref not in refered_xref:
                try:
                    img_pixmap = pymupdf.Pixmap(pdf_doc, xref)
                    pixmap_list.append(img_pixmap)
                    refered_xref.append(xref)
                except ValueError as e:
                    print(f"Skipping image with due to error: {e}")
        if len(pixmap_list) > 0:
            img_content = extract_image_content(
                pixmap_list=pixmap_list, text=text_content.replace("\n", "\t")
            )
            page_content = page_content + "\n\n" + "\n\n".join(img_content)

        pages_content.append(page_content)

    num_tokens = number_of_tokens(pages_content)

    final_data = []

    # Logic to handle case when page content > 512 tokens
    for e, n_token in enumerate(num_tokens):
        if n_token > 500:
            n_parts = numpy.ceil(n_token / 500).astype(int)
            len_content = len(pages_content[e])
            part_size = len_content // n_parts
            start, end = 0, part_size
            temp = []
            for _ in range(n_parts):
                temp.append((pages_content[e][start:end], e + 1))
                start = end
                end = end + part_size
            final_data += temp
        else:
            final_data.append((pages_content[e], e + 1))

    pdf_doc.close()
    print(final_data)
    return final_data


def markdown(output):
    report_html = output.get("report", "")
    references = output.get("references", {})
    references_markdown = ""

    for url, content in references.items():
        # Making the URL clickable in pure HTML
        clickable_url = f'<a href="{url}">{url}</a>'
        references_markdown += f"<details><summary>{clickable_url}</summary>\n\n{html2text.html2text(content)}</details>\n\n"

    combined_markdown = ""
    if report_html.strip():  # Check if report_html is not empty
        combined_markdown += html2text.html2text(report_html) + "\n\n"
    combined_markdown += references_markdown
    return combined_markdown


def pinecone_server():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "investment"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(1)
    index = pc.Index(index_name)
    index.describe_index_stats()
    return index


def fetch_vectorstore_from_db(file_id):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432",
    )
    cur = conn.cursor()
    create_table_query = """
        CREATE TABLE IF NOT EXISTS investment_research_pro (
            file_id VARCHAR(1024) PRIMARY KEY,
            file_name VARCHAR(1024),
            name_space VARCHAR(1024)

        );
    """
    cur.execute(create_table_query)
    conn.commit()
    fetch_query = """
    SELECT name_space
    FROM investment_research_pro
    WHERE file_id = %s;
    """
    cur.execute(fetch_query, (file_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0]
    return None


def get_next_namespace():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432",
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM investment_research_pro")
    count = cur.fetchone()[0]
    next_namespace = f"pdf-{count + 1}"
    cur.close()
    conn.close()
    return next_namespace


def insert_data(file_id, file_name, name_space):

    print("inserted")
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432",
    )
    cur = conn.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS investment_research_pro (
        file_id VARCHAR(1024) PRIMARY KEY,
        file_name VARCHAR(1024),
        name_space VARCHAR(255)
    );
    """
    cur.execute(create_table_query)
    conn.commit()
    insert_query = """
    INSERT INTO investment_research_pro (file_id, file_name, name_space)
    VALUES (%s, %s, %s)
    ON CONFLICT (file_id) DO NOTHING;
    """
    cur.execute(insert_query, (file_id, file_name, name_space))
    conn.commit()
    cur.close()
    conn.close()


def create_documents(page_contents):
    documents = []

    for content, page_number in page_contents:
        doc = {
            "page_content": content,
            "metadata": {"page_number": page_number, "original_content": content},
        }
        documents.append(doc)

    return documents


def embed_and_upsert(documents, name_space):
    chunks = [doc["page_content"] for doc in documents]
    pinecone_index = pinecone_server()
    embeddings_response = client.embeddings.create(
        input=chunks, model="BAAI/bge-large-en-v1.5"
    ).data
    embeddings = [i.embedding for i in embeddings_response]
    pinecone_data = []
    for doc, embedding in zip(documents, embeddings):
        i = str(uuid.uuid4())
        pinecone_data.append(
            {"id": i, "values": embedding, "metadata": doc["metadata"]}
        )

    pinecone_index.upsert(vectors=pinecone_data, namespace=name_space)


def embedding_creation(pdf_content, name_space):
    data = extract_content(pdf_content)
    # text_data = [i[0] for i in data]
    documents = create_documents(data)
    embed_and_upsert(documents, name_space)
    print("Embeddings created and upserted successfully into Pinecone.")


def embed(question):
    embeddings_response = client.embeddings.create(
        input=[question],
        model="BAAI/bge-large-en-v1.5",
    ).data
    embeddings = embeddings_response[0].embedding
    return embeddings


def process_rerank_response(rerank_response, docs):
    rerank_docs = []
    for item in rerank_response.results:
        index = item.index
        if 0 <= index < len(docs):
            rerank_docs.append(docs[index])
        else:
            print(f"Warning: Index {index} is out of range for documents list.")
    return rerank_docs


async def get_docs(question, pdf_content, file_name):
    global document_processing_event
    index = pinecone_server()
    co = cohere.Client(COHERE_API)
    xq = embed(question)

    await document_processing_event.wait()
    file_id = get_digest(pdf_content)
    existing_namespace = fetch_vectorstore_from_db(file_id)

    if existing_namespace:
        print("Document already exists. Using existing namespace.")
        name_space = existing_namespace
    else:
        document_processing_event.clear()
        print("evet stopped")
        print("Document is new. Creating embeddings and new namespace.")
        name_space = get_next_namespace()
        print(name_space)
        embedding_creation(pdf_content, name_space)
        insert_data(file_id, file_name, name_space)
        print("Sleep complete....")
        # except Exception as e:
        #    print(e)
        # finally:
        print("finally called")
        document_processing_event.set()

    # Query is now inside the lock to ensure it happens after any new document processing
    res = index.query(namespace=name_space, vector=xq, top_k=5, include_metadata=True)

    print(res)
    docs = [x["metadata"]["original_content"] for x in res["matches"]]

    if not docs:
        print("No matching documents found.")
        return []

    results = co.rerank(
        query=question, documents=docs, top_n=3, model="rerank-english-v3.0"
    )
    reranked_docs = process_rerank_response(results, docs)
    return reranked_docs


async def industry(pdf_content, file_name):
    question = (
        "What is the name and its specific niche business this document pertains to."
    )
    docs = await get_docs(question, pdf_content, file_name)
    context = "\n\n".join(docs)
    message = f"CONTEXT\n\n{context}\n\n"
    model = "meta-llama/llama-3-70b-instruct:nitro"
    response_str = response(
        message=message, model=model, SysPrompt=IndustryPrompt, temperature=0
    )
    industry = json.loads(response_str)
    print(industry)
    return industry


def split_into_chunks(input_string, token_limit=4500):
    # Initialize the tokenizer for the model
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Encode the input string to get the tokens
    tokens = encoding.encode(input_string)

    # List to store chunks
    chunks = []
    start = 0

    # Iterate over the tokens and split into chunks
    while start < len(tokens):
        end = start + token_limit
        if end >= len(tokens):
            chunk_tokens = tokens[start:]
        else:
            break_point = end
            while break_point > start and tokens[break_point] not in encoding.encode(
                " "
            ):
                break_point -= 1

            if break_point == start:
                chunk_tokens = tokens[start:end]
            else:
                chunk_tokens = tokens[start:break_point]
                end = break_point

        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
        start = end

    return chunks


def further_split_chunk(chunk, token_limit):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(chunk)
    sub_chunks = []
    start = 0

    while start < len(tokens):
        end = start + token_limit
        if end >= len(tokens):
            sub_chunk_tokens = tokens[start:]
        else:
            break_point = end
            while break_point > start and tokens[break_point] not in encoding.encode(
                " "
            ):
                break_point -= 1

            if break_point == start:
                sub_chunk_tokens = tokens[start:end]
            else:
                sub_chunk_tokens = tokens[start:break_point]
                end = break_point

        sub_chunk = encoding.decode(sub_chunk_tokens)
        sub_chunks.append(sub_chunk)
        start = end

    return sub_chunks


# Define the investment function
def investment(queries, query_results, other_info_results):
    time.sleep(1)
    # Combine queries and query_results into a dictionary
    combined_results = {q: r for q, r in zip(queries[-4:], query_results[-4:])}

    # Extract keys and answers from the other_info_results and update the combined_results dictionary
    for key, value in other_info_results.items():
        if isinstance(value, str):  # Check if the value is a string
            combined_results[key] = value.split("<details><summary>")[0].strip()
        else:
            combined_results[key] = value

    message = f"CONTEXT:\n\n{json.dumps(combined_results, indent=4)}\n\n"

    sys_prompt = Investment
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sys_prompt_token_size = len(encoding.encode(sys_prompt))

    max_model_tokens = 7000
    max_chunk_size = 7000  # Adjust to leave more buffer space

    chunks = split_into_chunks(message, token_limit=max_chunk_size)

    model = "meta-llama/llama-3-70b-instruct:nitro"
    responses = []
    tokens_used = 0
    max_tokens_per_minute = 6000

    for chunk in chunks:
        chunk_token_size = len(encoding.encode(chunk))
        combined_message = f"{sys_prompt}\n{chunk}"
        combined_token_size = len(encoding.encode(combined_message))

        print(
            f"Token size of the combined message and SysPrompt for this chunk: {combined_token_size}"
        )
        print(f"Chunk token size: {chunk_token_size}")
        print(f"SysPrompt token size: {sys_prompt_token_size}")

        if combined_token_size > max_model_tokens:
            print(
                f"Warning: Combined token size ({combined_token_size}) exceeds the model's limit ({max_model_tokens}). Adjusting chunk size."
            )
            sub_chunks = further_split_chunk(
                chunk, max_model_tokens - sys_prompt_token_size
            )
            for sub_chunk in sub_chunks:
                sub_chunk_token_size = len(encoding.encode(sub_chunk))
                print(sub_chunk_token_size)
                if sub_chunk_token_size > 500:
                    sub_combined_message = f"{sys_prompt}\n{sub_chunk}"
                    sub_combined_token_size = len(encoding.encode(sub_combined_message))
                    if sub_combined_token_size <= max_model_tokens:
                        response_str = response(
                            message=sub_chunk,
                            model=model,
                            SysPrompt=sys_prompt,
                            temperature=0,
                        )
                        print(response_str)
                        json_part = extract_json(response_str)
                        if json_part:
                            responses.append(json_part)
                        else:
                            print("Warning: No valid JSON part found in the response.")
                        tokens_used += sub_combined_token_size
                        if tokens_used >= max_tokens_per_minute:
                            print("Waiting for 60 seconds to avoid rate limit.")
                            time.sleep(60)
                            tokens_used = 0
        else:
            if chunk_token_size >= 500:
                response_str = response(
                    message=chunk, model=model, SysPrompt=sys_prompt, temperature=0
                )
                print(response_str)
                json_part = extract_json(response_str)
                if json_part:
                    responses.append(json_part)
                else:
                    print("Warning: No valid JSON part found in the response.")
                tokens_used += combined_token_size
                if tokens_used >= max_tokens_per_minute:
                    print("Waiting for 60 seconds to avoid rate limit.")
                    time.sleep(60)
                    tokens_used = 0

    combined_json = {"sectors": [], "final_score": 0}
    total_score = 0
    count = 0

    for response_str in responses:
        response_json = json.loads(response_str)
        combined_json["sectors"].append(response_json)
        total_score += response_json["overall_score"]
        count += 1

    if count > 0:
        combined_json["final_score"] = total_score / count
    final_json = json.dumps(combined_json, indent=4)
    print(final_json)
    return final_json


def extract_json(response_str):
    """Extract the JSON part from the response string."""
    match = re.search(r"\{.*}", response_str, re.DOTALL)
    if match:
        json_part = match.group()
        try:
            json.loads(json_part)  # Check if it's valid JSON
            return json_part
        except json.JSONDecodeError:
            print("Invalid JSON detected.")
    return None


async def answer(client, question, pdf_content, file_name):

    docs = await get_docs(question, pdf_content, file_name)
    context = "\n\n".join(docs)
    message = f"CONTEXT:\n\n{context}\n\nQUESTION :\n\n{question}\n\n"
    model = "meta-llama/llama-3-70b-instruct:nitro"
    messages = [
        {"role": "system", "content": QuestionRouter},
        {"role": "user", "content": message},
    ]
    response_str = await client.chat.completions.create(
        messages=messages, model=model, temperature=0
    )
    print(response_str)
    source = json.loads(response_str.choices[0].message.content)
    print(source)

    if source["datasource"].lower() == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        data_source = "vectorstore"
        message = f"CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}\n\nANSWER:\n"
        model = "meta-llama/llama-3-70b-instruct:nitro"
        messages = [
            {"role": "system", "content": GenerationPrompt},
            {"role": "user", "content": message},
        ]
        output = await client.chat.completions.create(
            messages=messages, model=model, temperature=0
        )

    elif source["datasource"].lower() == "missing_information":
        print("---NO SUFFICIENT INFORMATION---")
        data_source = "missing information"
        message = f"CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}\n\nANSWER:\n"
        model = "meta-llama/llama-3-70b-instruct:nitro"
        messages = [
            {"role": "system", "content": MissingInformation},
            {"role": "user", "content": message},
        ]
        output = await client.chat.completions.create(
            messages=messages, model=model, temperature=0
        )

    return output


async def process_queries(queries, pdf_content, file_name):
    # Run the `answer` function concurrently for all queries
    async_client = openai.AsyncClient(
        api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
    )
    async with async_client as aclient:
        tasks = [
            asyncio.create_task(answer(aclient, query, pdf_content, file_name))
            for query in queries
        ]
        responses = await asyncio.gather(*tasks)

    results = [response.choices[0].message.content for response in responses]
    return results


async def web_search(session, question):
    data = {
        "topic": "",
        "description": question,
        "user_id": "",
        "user_name": "",
        "internet": True,
        "output_format": "report_table",
        "data_format": "No presets",
    }
    async with session.post(
        "https://pvanand-search-generate-staging.hf.space/generate_report",
        json=data,
        headers={"Content-Type": "application/json"},
    ) as response:
        print(f"Status: {response.status}")
        print(f"Headers: {response.headers}")
        content = await response.text()
        print(f"Content: {content[:200]}...")  # Print first 200 chars of content
        if response.headers.get('Content-Type', '').startswith('application/json'):
            return await response.json()
        else:
            raise ValueError(f"Unexpected content type: {response.headers.get('Content-Type')}")


async def other_info(pdf_content, file_name):
    data = await industry(pdf_content, file_name)
    industry_company = data.get("industry")
    niche = data.get("niche")

    # Define the questions for each category
    questions = {
        "Risk Involved": f"What are risk involved in the starting a {niche} business in {industry_company}?",
        "Barrier To Entry": f"What are barrier to entry for a {niche} business in {industry_company}?",
        "Competitors": f"Who are the main competitors in the market for {niche} business in {industry_company}?",
        "Challenges": f"What are in the challenges in the {niche} business for {industry_company}?",
    }

    # Fetch the results for each category
    results = {}
    async with aiohttp.ClientSession() as session:
        tasks = [web_search(session, question) for question in questions.values()]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    for type_, response in zip(questions, responses):
        if isinstance(response, Exception):
            results[type_] = {"error": str(response)}
        else:
            results[type_] = markdown(response)

    return results

# Main function adapted for Streamlit
async def main(file_path, progress_callback=None):
    if not os.path.isfile(file_path):
        raise FileNotFoundError("File not found.")

    with open(file_path, "rb") as file:
        pdf_content = file.read()

    file_name = os.path.basename(file_path)

    # Process the queries
    if progress_callback:
        progress_callback("Answering queries from the pitch deck...", 33)
    query_results = await process_queries(queries, pdf_content, file_name)
    if progress_callback:
        progress_callback("", 0)

    # Process other information
    if progress_callback:
        progress_callback("Collecting other information...", 66)
    other_info_results = await other_info(pdf_content, file_name)
    if progress_callback:
        progress_callback("", 0)

    # Calculate grading results
    if progress_callback:
        progress_callback("Grading the results...", 100)
    grading_results = json.loads(investment(queries, query_results, other_info_results))
    print("\n\n\n", query_results, "\n\n\n")
    print(other_info_results, "\n\n\n")
    print(grading_results)
    if progress_callback:
        progress_callback("", 0)

    return queries, query_results, other_info_results, grading_results


# Run the main function
if __name__ == "__main__":
    file_path = input("Enter the path to the PDF file: ")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main(file_path))
