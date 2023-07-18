import streamlit as st
import pandas as pd
import numpy as np
import webbrowser
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the trained model and data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_name = pickle.load(open('artifacts/book_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

# Function to recommend books
def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)
    if len(book_id[0]) == 0:
        st.write("Book not found in the dataset.")
        return

    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id[0][0], :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = []
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        recommended_books.extend(books)

    return recommended_books

# Function to open Amazon Books search
def open_amazon_books_search(book_name):
    search_url = f"https://www.amazon.com/s?k={book_name.replace(' ', '+')}"
    webbrowser.open_new_tab(search_url)

# Set custom app theme
COLOR_PRIMARY = "#00b894"
COLOR_SECONDARY = "#dfe6e9"
BACKGROUND_COLOR = "#f1f2f6"
FONT_COLOR = "#333333"

st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styles
st.markdown(
    f"""
    <style>
    .stButton button {{
        color: {FONT_COLOR};
        background-color: {COLOR_PRIMARY};
        border-color: {COLOR_PRIMARY};
        font-size: 16px;
        padding: 0.5em 1em;
        border-radius: 0.5em;
        transition: all 0.3s ease-out;
    }}
    .stButton button:hover {{
        background-color: {COLOR_SECONDARY};
        color: {FONT_COLOR};
        cursor: pointer;
    }}
    .stButton button:active {{
        transform: translateY(1px);
    }}
    .book-title {{
        font-size: 24px;
        font-weight: bold;
        color: {FONT_COLOR};
        margin-bottom: 0.5em;
    }}
    .book-info {{
        font-size: 18px;
        color: {FONT_COLOR};
        margin-bottom: 0.5em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app
def main():
    st.title("BOOK RECOMMENDER SYSTEM USING MACHINE LEARNING")
    st.markdown("---")

    # Book selection dropdown
    book_selection = st.selectbox("Select a book:", book_name)

    if st.button("Recommend"):
        recommended_books = recommend_book(book_selection)

        st.subheader("Recommended Books:")

        # Display images in a row
        images_per_row = 3
        num_images = len(recommended_books)
        num_rows = (num_images // images_per_row) + (num_images % images_per_row > 0)

        for i in range(num_rows):
            row_start = i * images_per_row
            row_end = min((i + 1) * images_per_row, num_images)
            image_row = recommended_books[row_start:row_end]

            st.write('\n')

            for book in image_row:
                book_info = final_rating[final_rating['title'] == book]
                author = book_info['author'].values[0]
                year = book_info['year'].values[0]

                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.markdown(f'<p class="book-title">{book}</p>', unsafe_allow_html=True)
                    book_img_url = book_info['img_url'].values[0]
                    st.image(book_img_url, width=200)

                with col2:
                    st.markdown(f'<p class="book-info">Author: <strong>{author}</strong></p>', unsafe_allow_html=True)

                with col3:
                    st.markdown(f'<p class="book-info">Year: <strong>{year}</strong></p>', unsafe_allow_html=True)

                with col4:
                    # Button to search book on Amazon Books
                    button_key = f"search_button_{book}"  # Unique key for each button
                    st.button(
                        "Search on Amazon Books",
                        key=button_key,
                        on_click=open_amazon_books_search,
                        args=(book,),
                    )

            if i < num_rows - 1:
                st.markdown("---")

            st.write('\n')


# Run the app
if __name__ == '__main__':
    main()
