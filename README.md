# Data Query

Data Query is a natural language interface to query your PostgreSQL database using OpenAI's language models. Just type your question like "Show me all customers" and get SQL-generated answers instantly in table and chart form.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/data-query.git
cd data-query
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your-openai-api-key
```

## Connect Your Database

You can connect your PostgreSQL database in two ways:

### Option 1: Separate Fields

- Host
- Port
- Database
- Username
- Password

### Option 2: Full PostgreSQL URL

```
postgresql://username:password@localhost:5432/database_name
```

or

```
postgres://user:pass@hostname:5432/dbname
```

## Usage

### Run the App

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501)

### Ask Natural Language Questions

Examples:

- Show me all customers
- What’s the total revenue this month?
- Find the top 10 selling products

### See Instant Results

- Data in interactive tables
- Auto-generated SQL queries
- Recommended charts
- Download results as CSV

## Features

- AI-Powered: Converts natural language into SQL
- Smart Visualizations: Automatically recommends charts
- Secure: Blocks unsafe SQL keywords like DELETE, DROP, etc.
- Fast: Query caching and connection pooling
- Export: Download results as CSV

## Tech Stack

- Python 3.8+
- Streamlit (frontend)
- OpenAI GPT (gpt-4 / gpt-3.5-turbo)
- PostgreSQL
- SQLAlchemy (optional)
- dotenv, pandas, plotly

## Example Connection Strings

```
postgresql://username:password@localhost:5432/my_database
postgres://user:pass@remotehost.com:5432/dbname
```

## License

This project is licensed under the MIT License.

## Contributing

Pull requests and issues are welcome. Please feel free to contribute.
