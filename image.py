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
from tenacity import retry, stop_after_attempt, wait_random_exponential

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HELICON_API_KEY = os.getenv("HELICON_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API = os.getenv("COHERE_API")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

gemini.configure(api_key=GEMINI_API_KEY)

client = openai.OpenAI(
    base_url="https://api.together.xyz/v1", api_key=TOGETHER_API_KEY
)

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

2. **Provide Detailed Reasoning:**
   - **Qualitative Insights:**
     - Describe qualitative factors influencing each section's investment potential, such as market trends, competitive landscape, management quality, and strategic advantages.
   - **Quantitative Analysis:**
     - Include numerical data or statistical analysis to support your assessment, using metrics such as growth rates, financial ratios, market share, and other relevant financial indicators.

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
- Only use the KEYS (e.g., key1, key2) to identify the sections to be graded. Do not add or alter any sections beyond these specified keys. Grade only the sections that match the keys provided in the CONTEXT.

Take a deep breath and work on this problem step-by-step.

---
"""
ScoringPrompt = """

You are a seasoned venture capitalist with over 20 years of experience in evaluating pitch decks for startups. Your task is to meticulously analyze the provided CONTEXT, which contains various queries and their corresponding responses related to a pitch deck. Adopt a conservative approach to scoring, considering the inherent risks in investments. Assign lower scores to reflect potential concerns. Each topic should be analyzed, scored, and the output provided in a concise JSON format, including the final weighted score.

**Instructions:**

1. **Introduction and Context**:
   - **Context**: You will analyze a pitch deck containing multiple queries and their responses. These responses cover various aspects of the business, such as product/service details, target market, revenue streams, financial performance, team credentials, competitive landscape, and more.

2. **Scoring Methodology**:
   - **Weights and Scores**: Each query is associated with a weight that indicates its importance in the overall evaluation. Assign a score from 1 to 10 for each query response, where 1 indicates high risk or significant concerns and 10 indicates low risk or strong attributes.
     - **1-3**: High risk or severe weaknesses.
     - **4-6**: Moderate risk or noticeable weaknesses.
     - **7-9**: Low risk or minor weaknesses.
     - **10**: Very low risk or exceptional strengths.

3. **Analyze and Score Queries**:
   - **Query Analysis**: For each query, review the provided response, assign a score, and explain your reasoning. Use the provided weight to assess its impact on the overall evaluation.
   - **Focus Areas**:
     - **Clarity and Detail**: Assess whether the response is clear and detailed.
     - **Feasibility and Realism**: Consider the practicality and realism of the response.
     - **Risk and Uncertainty**: Evaluate potential risks and uncertainties highlighted in the response.
     - **Strengths and Weaknesses**: Identify any strengths or weaknesses in the response.

4. **Output Format**:
   - **JSON Object**: Each query, response, weight, score, and reasoning should be included in a concise JSON object. Additionally, the final weighted score should be included at the end. The JSON output should be structured as follows:
     ```json
     {
       "queries": [
         {
           "query": "What is the company's product/service, and what are its key features?",
           "response": "The product is a cloud-based software solution that offers real-time data analytics and reporting features.",
           "weight": 0.1,
           "score": 7,
           "reasoning": "Clear and detailed product description, but lacks differentiation from competitors."
         },
         {
           "query": "Who is the target customer for the company's product/service, and what problem does it solve for them?",
           "response": "The target customers are small to medium-sized businesses (SMBs) looking to improve their data management and reporting capabilities.",
           "weight": 0.1,
           "score": 6,
           "reasoning": "Well-defined target market, but lacks detail on the problem's significance."
         },
         // Continue for each query
       ],
       "total_weighted_score": 5.6 // Replace with calculated total score
     }
     ```

5. **Detailed Analysis and Scoring**:
   - **Review each query and response from the context**.
   - **For each query**, fill in the following template:

     ```json
     {
       "query": "[Insert query]",
       "response": "[Insert response from context]",
       "weight": [Insert weight],
       "score": [Assign a score],
       "reasoning": "[Provide a brief analysis of the response, highlighting any strengths, weaknesses, or risks that influenced your score.]"
     }
     ```

6. **Calculate Weighted Scores**:
   - **Weighted Score Calculation**: Multiply each score by its corresponding weight and sum these to get the total weighted score.
     \[
     \text{Total Score} = \sum (\text{Score}_i \times \text{Weight}_i)
     \]
   - **Add this total score to the JSON output**.

7. **Example Output**:
   - The final JSON should look like this:
     ```json
     {
       "queries": [
         {
           "query": "What is the company's product/service, and what are its key features?",
           "response": "The product is a cloud-based software solution that offers real-time data analytics and reporting features.",
           "weight": 0.1,
           "score": 7,
           "reasoning": "Clear and detailed product description, but lacks differentiation from competitors."
         },
         {
           "query": "Who is the target customer for the company's product/service, and what problem does it solve for them?",
           "response": "The target customers are small to medium-sized businesses (SMBs) looking to improve their data management and reporting capabilities.",
           "weight": 0.1,
           "score": 6,
           "reasoning": "Well-defined target market, but lacks detail on the problem's significance."
         },
         {
           "query": "What are the company's revenue streams?",
           "response": "The company generates revenue through subscription services, direct sales, and partnerships.",
           "weight": 0.1,
           "score": 5,
           "reasoning": "Positive diversification of revenue streams, but lacks detailed information on scale and stability."
         }
         // Add other queries similarly
       ],
       "total_weighted_score": 5.6 // Replace with calculated total score
     }
     ```

Take a deep breath and work on this problem step-by-step.
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
    "What issue or challenge is the company addressing?"

]


def get_digest(pdf_content):
    h = hashlib.sha256()
    h.update(pdf_content)  # Hash the binary content of the PDF
    return h.hexdigest()


def response(message: object, model: object = "llama3-8b-8192", SysPrompt: object = SysPromptDefault,
             temperature: object = 0.2) -> object:
    """
    :rtype: object
    """
    client = openai.OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://gateway.hconeai.com/openai/v1",
        default_headers={
            "Helicone-Auth": f"Bearer {HELICON_API_KEY}",
            "Helicone-Target-Url": "https://api.groq.com"
        }
    )

    messages = [{"role": "system", "content": SysPrompt}, {"role": "user", "content": message}]

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
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


def extract_image_content(pixmap: pymupdf.Pixmap, text: str) -> str:
    "Takes image path and extract information from it, and return it as text."

    description_prompt = f"You are provided with the images extracted from a pitch-deck and some text surrounding the image from the same pitch deck. Extract all the factual information that the image is trying to communicate through line charts, area line charts, bar charts, pie charts, tables exectra. Use OCR to extract numerical figures and include them in the information. If the image does not have any information like its a blank image or image of a person then response should be NOTHING. Do not add any additional comments or markdown, just give information. \n\n SURROUNDING TEXT \n\n{text}"

    model = gemini.GenerativeModel("gemini-1.5-flash")
    img = Image.frombytes(
        mode="RGB", size=(pixmap.width, pixmap.height), data=pixmap.samples
    )

    response = model.generate_content([description_prompt, img], stream=False)
    # print(response.text)  # just to visualize what model is generating, can be commented

    return response.text


def extract_content(pdf_content: bytes) -> List[Tuple[str, int]]:
    """
    Takes PDF(bytes) and return a list of tuples containing text(including textual and image content)
    and page number containing that text.
    """

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
        for img_info in image_list:
            xref = img_info["xref"]
            if xref not in refered_xref:
                # if xref not in refered_xref:
                try:
                    img_pixmap = pymupdf.Pixmap(pdf_doc, xref)
                    img_content = extract_image_content(
                        pixmap=img_pixmap, text=text_content.replace("\n", "\t")
                    )
                    page_content = page_content + "\n\n" + str(img_content)
                    refered_xref.append(xref)
                except ValueError as e:
                    print(f"Skipping image with due to error: {e}")

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
    report_html = output.get('report', '')
    references = output.get('references', {})
    references_markdown = ""

    for url, content in references.items():
        # Making the URL clickable in pure HTML
        clickable_url = f'<a href="{url}">{url}</a>'
        references_markdown += f'<details><summary>{clickable_url}</summary>\n\n{html2text.html2text(content)}</details>\n\n'

    combined_markdown = ""
    if report_html.strip():  # Check if report_html is not empty
        combined_markdown += html2text.html2text(report_html) + "\n\n"
    combined_markdown += references_markdown
    return combined_markdown


def pinecone_server():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'investment'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
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
        port="5432"
    )
    cur = conn.cursor()
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS investment_research_pro (
            file_id VARCHAR(1024) PRIMARY KEY,
            file_name VARCHAR(1024),
            name_space VARCHAR(1024)

        );
    '''
    cur.execute(create_table_query)
    conn.commit()
    fetch_query = '''
    SELECT name_space
    FROM investment_research_pro
    WHERE file_id = %s;
    '''
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
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM investment_research_pro")
    count = cur.fetchone()[0]
    next_namespace = f"pdf-{count + 1}"
    cur.close()
    conn.close()
    return next_namespace


def insert_data(file_id, file_name, name_space):
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres.kstfnkkxavowoutfytoq",
        password="nI20th0in3@",
        host="aws-0-us-east-1.pooler.supabase.com",
        port="5432"
    )
    cur = conn.cursor()
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS investment_research_pro (
        file_id VARCHAR(1024) PRIMARY KEY,
        file_name VARCHAR(1024),
        name_space VARCHAR(255)
    );
    '''
    cur.execute(create_table_query)
    conn.commit()
    insert_query = '''
    INSERT INTO investment_research_pro (file_id, file_name, name_space)
    VALUES (%s, %s, %s)
    ON CONFLICT (file_id) DO NOTHING;
    '''
    cur.execute(insert_query, (file_id, file_name, name_space))
    conn.commit()
    cur.close()
    conn.close()


def create_documents(page_contents):
    documents = []

    for content, page_number in page_contents:
        doc = {
            'page_content': content,
            'metadata': {
                'page_number': page_number,
                'original_content': content
            }
        }
        documents.append(doc)

    return documents


async def embed_and_upsert(documents, name_space):
    chunks = [doc['page_content'] for doc in documents]
    pinecone_index = pinecone_server()
    embeddings_response = client.embeddings.create(
        input=chunks, model="BAAI/bge-large-en-v1.5"
    ).data
    embeddings = [i.embedding for i in embeddings_response]
    pinecone_data = []
    for doc, embedding in zip(documents, embeddings):
        i = str(uuid.uuid4())
        pinecone_data.append({
            'id': i,
            'values': embedding,
            'metadata': doc['metadata']
        })

    pinecone_index.upsert(vectors=pinecone_data, namespace=name_space)


async def embedding_creation(pdf_content, name_space):
    data = extract_content(pdf_content)
    # text_data = [i[0] for i in data]
    documents = create_documents(data)
    await embed_and_upsert(documents, name_space)
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


async def get_name_space(question, pdf_content, file_name):
    file_id = get_digest(pdf_content)
    existing_namespace = fetch_vectorstore_from_db(file_id)

    if existing_namespace:
        print("Document already exists. Using existing namespace.")
        name_space = existing_namespace
    else:
        print("Document is new. Creating embeddings and new namespace.")
        name_space = get_next_namespace()
        await embedding_creation(pdf_content, name_space)
        insert_data(file_id, file_name, name_space)
        await asyncio.sleep(5)  # Use asyncio.sleep instead of time.sleep

    return name_space


async def get_docs(question, pdf_content, file_name):
    index = pinecone_server()
    co = cohere.Client(COHERE_API)
    xq = embed(question)
    name_space = await get_name_space(question, pdf_content, file_name)
    # print(name_space)
    res = index.query(namespace=name_space, vector=xq, top_k=5, include_metadata=True)
    print(res)
    docs = [x["metadata"]['original_content'] for x in res["matches"]]

    if not docs:
        print("No matching documents found.")
        return []

    results = co.rerank(query=question, documents=docs, top_n=3, model='rerank-english-v3.0')
    reranked_docs = process_rerank_response(results, docs)
    return reranked_docs


async def industry(pdf_content, file_name):
    question = "What is the name and its specific niche business this document pertains to."
    docs = await get_docs(question, pdf_content, file_name)
    context = "\n\n".join(docs)
    message = f"CONTEXT\n\n{context}\n\n"
    model = "llama3-70b-8192"
    response_str = response(message=message, model=model, SysPrompt=IndustryPrompt, temperature=0)
    industry = json.loads(response_str)
    print(industry)
    return industry


async def web_search(question):
    data = {
        "topic": "",
        "description": question,
        "user_id": "",
        "user_name": "",
        "internet": True,
        "output_format": "report_table",
        "data_format": "No presets",
    }
    response = requests.post(
        f"https://pvanand-search-generate-staging.hf.space/generate_report",
        json=data,
        headers={"Content-Type": "application/json"},
    )
    if response.status_code == 200:
        try:
            result = response.json()
            return result
        except requests.exceptions.JSONDecodeError:
            return {"error": "Failed to decode JSON response"}
    else:
        return {"error": f"Failed to ask question: {response.status_code}"}


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
            while break_point > start and tokens[break_point] not in encoding.encode(" "):
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


# Define the investment function
def investment(queries, query_results, other_info_results):
    # Combine queries and query_results into a dictionary
    combined_results = {q: r for q, r in zip(queries[-4:], query_results[-4:])}

    # Extract keys and answers from the other_info_results and update the combined_results dictionary
    for key, value in other_info_results.items():
        combined_results[key] = value.split('<details><summary>')[0].strip()

    message = f"CONTEXT:\n\n{json.dumps(combined_results, indent=4)}\n\n"
    # print(combined_results)
    sys_prompt = Investment
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    sys_prompt_token_size = len(encoding.encode(sys_prompt))

    max_model_tokens = 5500
    max_chunk_size = 5000  # Adjust to leave more buffer space

    chunks = split_into_chunks(message, token_limit=max_chunk_size)

    model = "llama3-70b-8192"
    responses = []
    tokens_used = 0
    max_tokens_per_minute = 5500

    for chunk in chunks:
        combined_message = f"{sys_prompt}\n{chunk}"
        combined_token_size = len(encoding.encode(combined_message))

        print(f"Token size of the combined message and SysPrompt for this chunk: {combined_token_size}")
        print(f"Chunk token size: {len(encoding.encode(chunk))}")
        print(f"SysPrompt token size: {sys_prompt_token_size}")

        if combined_token_size > max_model_tokens:
            print(
                f"Warning: Combined token size ({combined_token_size}) exceeds the model's limit ({max_model_tokens}). Adjusting chunk size.")

        response_str = response(message=chunk, model=model, SysPrompt=sys_prompt, temperature=0)
        # print(response_str)
        # Simulate response generation and track tokens used
        json_part = extract_json(response_str)
        if json_part:
            json_part = json_part
        else:
            print("Warning: No valid JSON part found in the response.")

        # Track tokens used and add delay to avoid rate limit
        tokens_used += combined_token_size
        print("Waiting for 60 seconds to avoid rate limit.")
        time.sleep(60)  # Wait for 60 seconds after each request

    final_json = json_part
    # print(final_json)
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


async def answer(question, pdf_content, file_name):
    docs = await get_docs(question, pdf_content, file_name)
    context = "\n\n".join(docs)
    message = f"CONTEXT:\n\n{context}\n\nQUESTION :\n\n{question}\n\n"
    model = "llama3-70b-8192"
    response_str = response(message=message, model=model, SysPrompt=QuestionRouter, temperature=0)
    print(response_str)
    source = json.loads(response_str)
    print(source)
    if source["datasource"].lower() == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        data_source = "vectorstore"
        message = f"CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}\n\nANSWER:\n"
        model = "llama3-70b-8192"
        output = response(message=message, model=model, SysPrompt=GenerationPrompt, temperature=0)
    elif source["datasource"].lower() == "missing_information":
        print("---NO SUFFICIENT INFORMATION---")
        data_source = "missing information"
        message = f"CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}\n\nANSWER:\n"
        model = "llama3-70b-8192"
        output = response(message=message, model=model, SysPrompt=MissingInformation, temperature=0)

    return output


async def process_queries(queries, pdf_content, file_name):
    results = []
    for query in queries:
        response = await answer(query, pdf_content, file_name)
        results.append((response))
    return results


def scoring_queries(queries, query_results):
    combined_results = {q: r for q, r in zip(queries[:-4], query_results[:-4])}
    message = f"CONTEXT:\n\n{json.dumps(combined_results, indent=4)}\n\n"
    model = "llama3-70b-8192"
    output = response(message=message, model=model, SysPrompt=ScoringPrompt, temperature=0)
    json_part = extract_json(output)
    final_json = json_part
    return final_json


def format_response_query(response):
    """Format the response appropriately for HTML output."""
    if isinstance(response, str):  # If the response is a plain string
        return f"<pre>{mistune.html(response)}</pre>"
    elif isinstance(response, list):  # If the response is a list
        return "<ul>" + "".join(f"<li>{item}</li>" for item in response) + "</ul>"
    elif isinstance(response, dict):  # If the response is a dictionary (JSON)
        return json.dumps(response, indent=4)  # Convert to formatted JSON string
    else:
        return str(response)


async def other_info(pdf_content, file_name):
    data = await industry(pdf_content, file_name)
    industry_company = data.get("industry")
    niche = data.get("niche")

    # Define the questions for each category
    questions = {
        "Risk Involved": f"What are risk involved in the starting a {niche} business in {industry_company}?",
        "Barrier To Entry": f"What are barrier to entry for a {niche} business in {industry_company}?",
        "Competitors": f"Who are the main competitors in the market for {niche} business in {industry_company}?",
        "Challenges": f"What are in the challenges in the {niche} business for {industry_company}?"
    }

    # Fetch the results for each category
    results = {}
    for category, question in questions.items():
        print(question)
        results[category] = markdown(await web_search(question))
    print(results)
    return results


def save_to_html(queries, query_results, grading_results, other_info_results, output_file):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Query Responses</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                background-color: #f9f9f9;
                font-size: 18px; /* Increase font size */
            }
            h1 {
                font-size: 40px; /* Increase font size */
                color: black;
                font-weight: 500;
                padding: 0px 0px 15px 0px;
                text-align: center;
                margin-bottom: 40px;
            }
            h2 {
                font-size: 30px; /* Increase font size */
                color: black;
                padding: 10px 0px 10px 0px;
                border-bottom: 2px solid #4CAF50;
                margin-top: 40px;
            }
            h3 {
                color: black;
                font-size: 28px; /* Increase font size */
                font-weight: 500;
                padding: 10px 0px 10px 0px;
            }
            h4 {
                color: black;
                font-size: 24px; /* Increase font size */
                padding: 5px 0px 5px 0px;
            }
            p {
                color: black;
                padding: 5px 0px 5px 0px;
                font-size: 18px; /* Increase font size */
            }
            strong {
                font-weight: 500;
            }
            li {
                color: black;
                display: flex;
                gap: 10px;
                padding: 0px 0px 0px 15px;
                font-size: 18px; /* Increase font size */
            }
            li strong {
                min-width: 300px;
                margin-right: 10px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin-top: 20px;
                font-size: 18px; /* Increase font size */
            }
            th, td {
                text-align: left;
                padding: 10px;
                border: 1px solid #ddd;
                vertical-align: top;
            }
            th {
                background-color: #4CAF50;
                color: white;
                text-align: center;
                font-size: 1.5em; /* Increase font size */
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            td {
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            tr {
                border-bottom: 1px solid #D4D4D4;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .summary {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .response {
                white-space: pre-wrap;
            }
            ul {
                display: flex;
                flex-direction: column;
            }
            .links {
                display: flex;
                flex-direction: column;
            }
            summary {
                width: 100%;
                border: 1px solid #ddd;
                margin-bottom: 10px;
                background-color: #ffffff;
                padding: 10px;
                border-radius: 8px;
            }
            .section-divider {
                border-top: 2px solid #4CAF50;
                margin: 40px 0;
            }
        </style>
    </head>
    <body style="page-break-before: always;">
        <h1>Query Responses</h1>
        <table>
            <tr>
                <th class="question">Query</th>
                <th>Answer</th>
            </tr>
    """

    # Add query responses
    for query, response in zip(queries, query_results):
        html_response = format_response_query(response)
        html_content += f"""
            <tr>
                <td class="question">{query}</td>
                <td class="response">{html_response}</td>
            </tr>
        """

    html_content += """
        </table>
    """

    # Add other information sections
    for category, response in other_info_results.items():
        html_response = format_response_query(response)
        html_content += f"""
        <div class="section-divider"></div>
        <h2>{category}</h2>
        <div class="response">{html_response}</div>
        """

    # Add grading results table
    html_content += """
        <div class="section-divider"></div>
        <h2>Grading Results</h2>
        <table>
            <tr>
                <th>Section</th>
                <th>Score</th>
                <th>Weight</th>
                <th>Reasoning</th>
            </tr>
    """

    for section_data in grading_results['sections']:
        html_content += f"""
            <tr>
                <td>{section_data['section']}</td>
                <td>{section_data['score']}</td>
                <td>{section_data['weight']}</td>
                <td>{section_data['reasoning']}</td>
            </tr>
        """

    html_content += """
        </table>
    """

    html_content += f"""
        <div class="section-divider"></div>
        <h3>Overall Score</h3>
        <p>{grading_results['overall_score']}</p>
    """

    html_content += """
        </body>
        </html>
        """

    # Write HTML content to file
    with open(output_file, 'w') as file:
        file.write(html_content)
    return output_file


def convert_html_to_pdf(html_file, pdf_file):
    options = {
        "enable-local-file-access": "",
        'encoding': "UTF-8",
        'print-media-type': '',
        'page-width': '3000px',  # Increase page width to accommodate horizontal scroll
        'page-height': '1000px', # Adjust height as necessary
        'margin-top': '5mm',
        'margin-bottom': '5mm',
        'margin-left': '5mm',
        'margin-right': '5mm',
        'zoom': '1.5'
    }
    try:
        pdfkit.from_file(html_file, pdf_file, options=options)
    except IOError as e:
        pass


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
    if progress_callback:
        progress_callback("", 0)



    return queries, query_results, other_info_results, grading_results


# Run the main function
if __name__ == "__main__":
    file_path = input("Enter the path to the PDF file: ")
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main(file_path))




