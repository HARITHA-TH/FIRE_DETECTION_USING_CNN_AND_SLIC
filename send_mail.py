import smtplib, ssl

smtp_server = "smtp.gmail.com"
port = 587  # For starttls
sender_email = "harithath003@gmail.com"
password = 'Chikku@123'


receiver_email = "harithath003@gmail.com"


def send():
    # Create a secure SSL context
    context = ssl.create_default_context()
    message = """Subject: FIRE ALERT \n\n\
    
    A fire is detected in nearby place....
    
    """

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitte d
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit() 