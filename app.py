import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# import torch


def getLLamares(input_txt,no_words,blog_style):

    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    llm = CTransformers(model="/Users/amitsarang/Downloads/marx-3b.ggmlv3.q4_0.bin",
                        model_type = 'llama',
                        config={'max_new_tokens':256,
                                'temperature':0.02})
    
    template = '''
        Write a blog for {blog_style} job profile for a topic {input_txt}
                within {no_words} words.
        '''
    prompt=PromptTemplate(input_variables=["blog_style","input_txt","no_words"],
                        template=template)
    
    response = llm(prompt.format(blog_style=blog_style,input_txt=input_txt,no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Genarating Blogs",
                page_icon='🤖',
                layout='centered',
                initial_sidebar_state='collapsed')

st.header("Blog Generater!")
input_txt = st.text_input("Enter the Blog topic")

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input("Number of words")
with col2:
    blog_style = st.selectbox("Writing blog for",
                            ("researchers","engineers","common people","children"),index=0)

submit=st.button("Generate")

if submit:
    st.write(getLLamares(input_txt,no_words,blog_style))        

