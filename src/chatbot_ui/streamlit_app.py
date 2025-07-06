import streamlit as st
import requests

from src.chatbot_ui.core.config import settings

st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}


if "retrieved_items" not in st.session_state:
    st.session_state.retrieved_items = []

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]

if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

if "sidebar_key" not in st.session_state:
    st.session_state.sidebar_key = 0

if "sidebar_placeholder" not in st.session_state:
    st.session_state.sidebar_placeholder = None

# Sidebar - Suggestions
with st.sidebar:
    st.markdown("### Suggestions")
    
    # Create or get the placeholder
    if st.session_state.sidebar_placeholder is None:
        st.session_state.sidebar_placeholder = st.empty()
    
    # Clear and rebuild the suggestions
    with st.session_state.sidebar_placeholder.container():
        if st.session_state.retrieved_items:
            for idx, item in enumerate(st.session_state.retrieved_items):
                st.divider()
                st.caption(item.get('description', 'No description'))
                if 'image_url' in item:
                    st.image(item["image_url"], width=300)
                st.caption(f"Price: {item['price']} USD")
        else:
            st.info("No suggestions yet")

# Main content - Chat interface

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):
        status, output = api_call("post", f"{settings.API_URL}/rag", json={"query": prompt})
        # Update retrieved items
        st.session_state.retrieved_items = output.get("used_image_urls", [])
        
        # Clear the sidebar placeholder to force refresh
        if st.session_state.sidebar_placeholder is not None:
            st.session_state.sidebar_placeholder.empty()
        
        response_content = output.get("answer", str(output))
    
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.rerun()