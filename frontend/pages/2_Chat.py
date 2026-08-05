"""Chat: conversational meal planning with the local LLM."""

from __future__ import annotations

import streamlit as st

import api_client as api
from components import chat_message
from utils import helpers

helpers.setup_page("Chat", icon="💬")
user_id = helpers.render_sidebar()

st.markdown("# 💬 Meal Planning Chat")
st.caption("Ask for a meal plan based on your remaining macros. "
           "Try: *“What should I have for dinner?”* or *“give me a grocery list for meal prep”*.")

# Render history.
for i, msg in enumerate(st.session_state.chat_history):
    chat_message.chat_bubble(msg["role"], msg["content"])
    if msg["role"] == "assistant":
        chat_message.meal_plan_card(msg.get("meal_plan"), key=f"hist_{i}")
        if msg.get("grocery_list_id"):
            st.success(f"🛒 A grocery list (#{msg['grocery_list_id']}) was created — see the Grocery page.")

prompt = st.chat_input("Message Macromancer…")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    chat_message.chat_bubble("user", prompt)
    with st.spinner("🧠 Thinking… (the local LLM can take a few seconds)"):
        try:
            res = api.chat(user_id, prompt, st.session_state.chat_session_id)
            st.session_state.chat_session_id = res.get("session_id")
            gl_id = res.get("grocery_list_id")
            if gl_id and gl_id not in st.session_state.grocery_list_ids:
                st.session_state.grocery_list_ids.append(gl_id)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": res.get("message", ""),
                "meal_plan": res.get("meal_plan"),
                "grocery_list_id": gl_id,
            })
        except api.APIError as exc:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ {exc}"}
            )
    st.rerun()

if st.session_state.chat_history and st.button("🧹 Clear conversation"):
    st.session_state.chat_history = []
    st.session_state.chat_session_id = None
    st.rerun()
