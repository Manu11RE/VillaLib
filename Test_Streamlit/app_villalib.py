import streamlit as st

# Simulación de una base de datos de libros
books_db = {
    '001': {'title': '1984', 'author': 'George Orwell', 'year': 1949, 'summary': 'Dystopian novel set in a totalitarian society.'},
    '002': {'title': 'To Kill a Mockingbird', 'author': 'Harper Lee', 'year': 1960, 'summary': 'A novel about racial injustice in the Deep South.'},
    '003': {'title': 'Pride and Prejudice', 'author': 'Jane Austen', 'year': 1813, 'summary': 'A classic novel of manners and romance.'}
}


# Función para mostrar los libros disponibles
def display_books():
    st.subheader("Libros Disponibles")
    for book_id, details in books_db.items():
        st.write(f"ID: {book_id}")
        st.write(f"Título: {details['title']}")
        st.write(f"Autor: {details['author']}")
        st.write(f"Año: {details['year']}")
        st.write("---")

# Función para consultar un libro por ID
def consult_book(book_id):
    st.subheader("Consulta de Libro")
    book = books_db.get(book_id, None)
    if book:
        st.write(f"Título: {book['title']}")
        st.write(f"Autor: {book['author']}")
        st.write(f"Año: {book['year']}")
        st.write(f"Resumen: {book['summary']}")
    else:
        st.write("Libro no encontrado.")

# Sección del chat
def chat_section():
    st.subheader("Chat con IA")
    user_input = st.text_input("Escribe tu mensaje aquí")
    if st.button("Enviar"):
        st.write(f"Tú: {user_input}")
        # Aquí se incluiría la respuesta del modelo de IA, pero por ahora dejaremos una respuesta genérica
        st.write("IA: (Respuesta del modelo de IA aquí)")

# Streamlit app
st.title("Aplicación de Biblioteca con Chat IA")

# Sidebar para la navegación
st.sidebar.title("Navegación")
section = st.sidebar.radio("Ir a", ("Chat con IA", "Ver Libros Disponibles", "Consultar Libro"))

if section == "Chat con IA":
    chat_section()
elif section == "Ver Libros Disponibles":
    display_books()
elif section == "Consultar Libro":
    book_id = st.text_input("Ingresa el ID del libro que quieres consultar")
    if st.button("Consultar"):
        consult_book(book_id)
