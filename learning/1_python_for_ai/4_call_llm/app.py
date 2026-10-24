import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

def generate_recipe(ingredients, cuisine, diet):
    prompt = f'''
    Generate one food recipe using these ingredients: {ingredients}
    Recipe should not be more than 100 words
    Cuisine: {cuisine}
    Diet: {diet}
    '''
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


st.title("Recipe Generator")

ingredients = st.text_area("Ingredients", placeholder="e.g. tomatoes, lentils, onion, cumin, salt")

diet = st.selectbox("Diet", ["Any", "Vegan", "Jain", "Vegetarian"])

cuisine = st.selectbox("Cuisine", ["Any", "American", "Italian", "Thai", "Mexican", "Indian", "Chinese"])

if st.button("Generate Recipe"):
    if not ingredients.strip():
        st.warning("Please enter some ingredients.")
    else:
        with st.spinner("Generating recipe..."):
            recipe = generate_recipe(ingredients, cuisine, diet)
        st.markdown(recipe)
