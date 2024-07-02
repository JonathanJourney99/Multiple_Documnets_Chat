css = """
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #1D2B53
}
.chat-message.bot {     
    background-color: #0F67B1
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
""" 

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://p7.hiclipart.com/preview/783/780/16/telegram-bot-api-security-token-internet-bot-chatbot-robot.jpg">
    </div>
    <div class="message">{{MSG}}</div>  
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://tse2.mm.bing.net/th?id=OIP.i1uuZ7_MdpdYue7bKIntuwAAAA&pid=Api&P=0&h=180   ">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
""" 