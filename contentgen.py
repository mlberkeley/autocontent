import argparse
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

def main(general_context, context, source_file, output_type):
    model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)

    chunk_size = 4000
    overlap = 500
    text_splitter = CharacterTextSplitter(separator=".", chunk_size=chunk_size, chunk_overlap=overlap)

    with open(source_file) as f:
        content = f.read()

    text_chunks = text_splitter.split_text(content)
    docs = [Document(page_content=t) for t in text_chunks]

    if output_type == 'blog':
        template = f"""{general_context} Write part of a detailed blog post based on the following text generated from {context}:\n\n{"{text}"}\n\nBLOG POST:"""
    else:
        template = f"""{general_context} Write a factual, dense knowledge article for an internal wiki summarizing the following text generated from {context}:\n\n{"{text}"}\n\nARTICLE:"""

    prompt = PromptTemplate(template=template, input_variables=["text"])
    chain = load_summarize_chain(model, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, return_intermediate_steps=True)
    output = chain({"input_documents": docs})

    with open("output.txt", "w+") as f:
        f.write(output['output_text'])

    with open("steps.txt", "w+") as f:
        f.write(str(output['intermediate_steps']))

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary from a given document")
    parser.add_argument("source_file", help="The source text document")
    parser.add_argument("--general-context", default="You are developing content for a student organization at UC Berkeley called ML@B that specializes in machine learning.", help="The context to generate the summary")
    parser.add_argument("--context", default="a talk about a recent paper titled ChatGPT as a Data Scientist: Text Mining Tasks for Language Models", help="The context to generate the summary")
    parser.add_argument("--output_type", choices=["blog", "article"], default="blog", help="Select output type: blog or article")

    args = parser.parse_args()
    main(args.general_context, args.context, args.source_file, args.output_type)