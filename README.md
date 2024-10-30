# Movie Review Sentiments
A streamlit app that determines whether a text field is positive or negative. This app implements a Logarithmic Regression model trained on 50,000 IMDB Reviews. In testing, this model is 90% accurate on determing if a movie review is positive or negative.
## Usage
To view the deployed version on streamlit, click the link on the sidebar  
  
To run locally:  
Download all files and dependencies, and navigate to the folder on the command line.  
Type the following into the command line:
```bash
streamlit run app.py
```
This will automatically open the streamlit app in your browser.  
Here, you can enter in text and the model can predict whether it is positive or negative.  
## Dependencies
To run this project, you need to install the following Python packages:

- pandas
- scikit-learn
- nltk
- beautifulsoup4
- streamlit
- joblib

You can install the required packages using pip:
```bash
pip install pandas scikit-learn nltk beautifulsoup4 streamlit joblib
```
## Authors
Dante Paradis - danteparadis77@gmail.com
## License
This project is licensed under the MIT License - see the LICENSE.md file for details
## Acknowledgments
Dataset - https://ai.stanford.edu/~amaas/data/sentiment/
