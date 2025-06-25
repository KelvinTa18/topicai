How to Set Up and Run the Application
# 1. Download All Required Files and Folders

  Download all files and folders provided for this project to your local machine.

# 2. Install Python 3.13

  Download and install Python 3.13 from the official Python website.

# 3. Install Project Dependencies
        
  Open your command line or terminal.
  Navigate to the project directory (where requirements.txt is located).
  
  Run the following command:
  
          pip install -r requirements.txt

# 4. Download Required NLTK Data

  Open Python and run the following code:
  
          import nltk
          nltk.download('punkt')
          nltk.download('stopwords')

# 5. Run the Django Application

  In your project directory, run:
        
        python manage.py runserver
        
  The application will be available at http://127.0.0.1:8000/ by default.
