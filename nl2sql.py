from openai import OpenAI
import re
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import json
import os
from dotenv import load_dotenv
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

class NL2SQLConverter:
    """
    Enhanced Natural Language to SQL converter using OpenAI's GPT models.
    Features: Security, caching, better error handling, and schema-aware suggestions.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the converter with API key and security settings."""
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
            
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Model fallback chain - try from best to most basic
        self.models = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
        self.current_model = None
        
        # Query cache for performance
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Security: Only allow safe, read-only operations
        self.forbidden_keywords = {
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE',
            'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'START', 'BEGIN', 'END',
            'DECLARE', 'SET', 'CALL', 'EXECUTE', 'EXEC', 'MERGE', 'REPLACE'
        }
        
        # Initialize the AI model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the first available OpenAI model with retry logic."""
        for attempt in range(3):  # 3 attempts
            for model_name in self.models:
                try:
                    # Test the model with a simple request
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "Respond with: OK"}],
                        max_tokens=10,
                        temperature=0
                    )
                    
                    if response and response.choices and "OK" in response.choices[0].message.content:
                        self.current_model = model_name
                        logger.info(f"Successfully initialized model: {model_name}")
                        return
                except Exception as e:
                    logger.warning(f"Failed to initialize model {model_name} (attempt {attempt + 1}): {e}")
                    continue
            
            if attempt < 2:  # Wait before retry
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("Failed to initialize any OpenAI model after 3 attempts. Check your API key and network connection.")

    def _get_cache_key(self, natural_query: str, schema_data: List[Dict[str, Any]]) -> str:
        """Generate cache key for query."""
        schema_hash = hash(str(sorted([table.get('table_name', '') for table in schema_data])))
        return f"{hash(natural_query.lower().strip())}_{schema_hash}"

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl


    def _validate_sql_query(self, query: str) -> Tuple[bool, str]:
        """Enhanced SQL query validation with better error messages."""
        if not query or not query.strip():
            return False, "❌ Query is empty"

        # Remove comments to avoid bypassing security checks
        clean_query = re.sub(r'--.*', '', query, flags=re.MULTILINE)
        clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
        clean_query = ' '.join(clean_query.split()).upper()
        logger.debug(f"Cleaned query for validation: {clean_query}")

        # Check for forbidden keywords - STRICT SECURITY CHECK
        for keyword in self.forbidden_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, clean_query):
                logger.warning(f"Security violation detected: forbidden keyword '{keyword}' found in query")
                return False, "🚫 No DDL or DML queries are allowed. Only SELECT queries are supported."

        # Must start with SELECT or WITH (for CTEs)
        if not re.match(r'^\s*(SELECT|WITH)\b', clean_query):
            return False, "❌ Query must start with SELECT or WITH statement"

        # Basic syntax checks
        if clean_query.count('(') != clean_query.count(')'):
            return False, "❌ Syntax Error: Unmatched parentheses in query"

        # Check for common SQL injection patterns
        injection_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT)',
            r'UNION\s+SELECT.*--',
            r'OR\s+1\s*=\s*1',
            r'AND\s+1\s*=\s*1'
        ]
        for pattern in injection_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return False, "🚫 Security Error: Potential SQL injection detected"

        # If all checks pass, the query is considered valid
        return True, "✅ Query is valid and safe" # <-- ADD THIS LINE


    def _create_schema_context(self, schema_data: List[Dict[str, Any]]) -> str:
        """Enhanced schema description with better formatting and examples."""
        if not schema_data:
            return "No schema information available."
        
        schema_lines = ["=== DATABASE SCHEMA ==="]
        total_tables = len(schema_data)
        total_columns = sum(len(table.get('columns', [])) for table in schema_data)
        
        schema_lines.append(f"Database contains {total_tables} tables with {total_columns} total columns\n")
        
        for table in schema_data:
            table_name = table.get('table_name', 'unknown_table')
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append("=" * (len(table_name) + 10))
            
            columns = table.get('columns', [])
            if not columns:
                schema_lines.append("  No column information available")
                continue
            
            # Group columns by type for better understanding
            key_columns = []
            text_columns = []
            numeric_columns = []
            date_columns = []
            other_columns = []
            
            for column in columns:
                col_name = column.get('column_name', 'unknown')
                col_type = column.get('data_type', 'unknown')
                
                if 'id' in col_name.lower() or col_name.lower().endswith('_id'):
                    key_columns.append(column)
                elif col_type in ['text', 'varchar', 'char', 'character varying']:
                    text_columns.append(column)
                elif col_type in ['integer', 'bigint', 'decimal', 'numeric', 'real', 'double precision']:
                    numeric_columns.append(column)
                elif col_type in ['date', 'timestamp', 'timestamptz', 'time']:
                    date_columns.append(column)
                else:
                    other_columns.append(column)
            
            # Display columns by category
            for category, cols in [
                ("Key Columns", key_columns),
                ("Text Columns", text_columns), 
                ("Numeric Columns", numeric_columns),
                ("Date/Time Columns", date_columns),
                ("Other Columns", other_columns)
            ]:
                if cols:
                    schema_lines.append(f"  {category}:")
                    for column in cols:
                        col_desc = self._format_column_description(column)
                        schema_lines.append(f"    • {col_desc}")
            
            schema_lines.append("")  # Empty line between tables
        
        return '\n'.join(schema_lines)

    def _format_column_description(self, column: Dict) -> str:
        """Format individual column description with enhanced details."""
        col_name = column.get('column_name', 'unknown')
        col_type = column.get('data_type', 'unknown')
        is_nullable = column.get('is_nullable', 'YES')
        default_val = column.get('column_default')
        
        col_desc = f"{col_name} ({col_type})"
        
        # Add length info for varchar/char
        if column.get('character_maximum_length'):
            col_desc = col_desc.replace(f"({col_type})", f"({col_type}({column['character_maximum_length']}))")
        
        # Add precision for numeric
        if column.get('numeric_precision'):
            precision = column['numeric_precision']
            scale = column.get('numeric_scale', 0)
            col_desc = col_desc.replace(f"({col_type})", f"({col_type}({precision},{scale}))")
        
        if is_nullable == 'NO':
            col_desc += " [NOT NULL]"
        
        if default_val:
            col_desc += f" [DEFAULT: {default_val}]"
        
        # Add sample values hint if available
        sample_values = column.get('sample_values', [])
        if sample_values:
            col_desc += f" [Examples: {', '.join(map(str, sample_values[:3]))}]"
        
        return col_desc

    def _build_enhanced_prompt(self, natural_query: str, schema_context: str) -> str:
        """Build enhanced prompt with better instructions and examples."""
        return f"""You are an expert PostgreSQL database assistant specialized in converting natural language to SQL queries.

PRIMARY OBJECTIVES:
1. Generate ONLY valid PostgreSQL SELECT queries
2. Use proper PostgreSQL syntax, functions, and best practices
3. Always add appropriate LIMIT clauses (default: 1000) unless user specifies otherwise
4. Use meaningful column aliases for better readability
5. Handle text searches with case-insensitivity using ILIKE

STRICT SECURITY RULES:
- ONLY SELECT and WITH statements allowed
- NO INSERT, UPDATE, DELETE, DROP, CREATE, ALTER operations
- NO stored procedures or function calls that modify data
- Always validate input parameters

SQL BEST PRACTICES:
✅ Use ILIKE for case-insensitive text matching: WHERE name ILIKE '%john%'
✅ Use proper JOINs when working with multiple tables
✅ Add meaningful aliases: SELECT customer_name AS "Customer Name"
✅ Use appropriate aggregate functions: COUNT(), SUM(), AVG(), MAX(), MIN()
✅ Sort results logically: ORDER BY relevant_column DESC/ASC
✅ Group data when using aggregates: GROUP BY necessary_columns

TEXT SEARCH EXAMPLES:
- "find customers with paypal": WHERE payment_method ILIKE '%paypal%'
- "orders from john": WHERE customer_name ILIKE '%john%'
- "active status": WHERE status ILIKE '%active%'

QUERY INTELLIGENCE:
- If query is ambiguous, make reasonable assumptions based on schema
- For "top N" requests, use ORDER BY with LIMIT
- For counts, use COUNT(*) or COUNT(column_name)
- For trends, consider date-based grouping and ordering

{schema_context}

USER QUESTION: {natural_query}

IMPORTANT: Return ONLY the SQL query - no explanations, markdown formatting, or code blocks.

GENERATED POSTGRESQL QUERY:"""

    def convert_to_sql(self, natural_query: str, schema_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced SQL conversion with caching and better error handling."""
        try:
            if not self.current_model:
                logger.warning("Model not initialized, attempting to reinitialize...")
                self._initialize_model()
            
            if not natural_query.strip():
                return {
                    'success': False,
                    'error': '❌ Natural language query cannot be empty',
                    'sql_query': None,
                    'suggestions': self.get_sample_queries_from_schema(schema_data)
                }
            
            # Early check: does the natural language query directly mention DDL/DML operations?
            if re.search(r'\b(drop|insert|update|delete|truncate|alter|create|grant|revoke|merge|replace)\b', natural_query, re.IGNORECASE):
                logger.warning(f"DDL/DML operation detected in natural query: {natural_query}")
                return {
                    'success': False,
                    'error': '🚫 No DDL or DML queries are allowed. Only SELECT queries are supported.',
                    'sql_query': None,
                    'suggestions': self.get_sample_queries_from_schema(schema_data)
                }
            
            # Check cache first
            cache_key = self._get_cache_key(natural_query, schema_data)
            if cache_key in self.query_cache and self._is_cache_valid(self.query_cache[cache_key]):
                logger.info("Returning cached SQL query")
                cached_result = self.query_cache[cache_key]['result'].copy()
                cached_result['from_cache'] = True
                return cached_result
            
            # Create schema context and prompt
            schema_context = self._create_schema_context(schema_data)
            prompt = self._build_enhanced_prompt(natural_query, schema_context)
            
            # Try to generate SQL with retries
            max_attempts = 3
            last_error = None
            generated_queries = []
            
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Generating SQL query (attempt {attempt + 1}/{max_attempts})")
                    
                    # Generate content with OpenAI
                    response = self.client.chat.completions.create(
                        model=self.current_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.1,  # Lower temperature for consistency
                        top_p=0.8,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    
                    if not response or not response.choices or not response.choices[0].message.content:
                        last_error = f"Empty response from model on attempt {attempt + 1}"
                        logger.warning(last_error)
                        continue
                    
                    # Enhanced query cleaning
                    sql_query = self._clean_sql_response(response.choices[0].message.content)
                    generated_queries.append(sql_query)
                    
                    logger.info(f"Generated query on attempt {attempt + 1}: {sql_query}")
                    
                    # Validate immediately - STOP EARLY on security violations
                    is_valid, validation_message = self._validate_sql_query(sql_query)
                    
                    # Stop immediately if query contains forbidden operations (security violation)
                    if not is_valid and ("Security Error" in validation_message or "No DDL or DML" in validation_message):
                        logger.warning(f"Security violation detected in generated SQL: {sql_query}")
                        return {
                            'success': False,
                            'error': '🚫 No DDL or DML queries are allowed. Only SELECT queries are supported.',
                            'sql_query': None,
                            'validation_message': validation_message,
                            'attempts_used': attempt + 1,
                            'from_cache': False
                        }
                    
                    if is_valid:
                        # Ensure LIMIT is present for SELECT queries
                        if (sql_query.strip().upper().startswith('SELECT') and 
                            'LIMIT' not in sql_query.upper() and
                            'COUNT(' not in sql_query.upper()):
                            sql_query = sql_query.rstrip(';') + ' LIMIT 1000'
                        
                        result = {
                            'success': True,
                            'sql_query': sql_query,
                            'validation_message': validation_message,
                            'attempts_used': attempt + 1,
                            'from_cache': False
                        }
                        
                        # Cache the successful result
                        self.query_cache[cache_key] = {
                            'result': result.copy(),
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"Successfully generated valid SQL query: {sql_query}")
                        return result
                    else:
                        last_error = validation_message
                        logger.warning(f"Invalid SQL on attempt {attempt + 1}: {validation_message}")
                        # Enhance prompt with specific feedback
                        prompt += f"\n\n🚫 CORRECTION NEEDED:\nPrevious attempt failed: {validation_message}\nGenerate a CORRECTED query following PostgreSQL syntax exactly."
                
                except Exception as e:
                    last_error = f"Error during SQL generation attempt {attempt + 1}: {str(e)}"
                    logger.error(last_error)
                    continue
            
            # Return detailed error with suggestions
            return {
                'success': False,
                'error': f'❌ Failed to generate valid SQL after {max_attempts} attempts.\n\n🔍 Last error: {last_error}',
                'sql_query': generated_queries[-1] if generated_queries else None,
                'attempts_used': max_attempts,
                'generated_queries': generated_queries,
                'suggestions': self.get_sample_queries_from_schema(schema_data),
                'natural_query': natural_query
            }
        
        except Exception as e:
            logger.error(f"Critical error in convert_to_sql: {e}")
            return {
                'success': False,
                'error': f' Critical conversion error: {str(e)}',
                'sql_query': None,
                'suggestions': self.get_sample_queries_from_schema(schema_data)
            }

    def _clean_sql_response(self, response_text: str) -> str:
        """Enhanced SQL response cleaning with better pattern matching."""
        sql_query = response_text.strip()
        
        # Remove various markdown formatting
        sql_query = re.sub(r'```sql\s*\n?|```\s*\n?|```', '', sql_query, flags=re.IGNORECASE).strip()
        sql_query = re.sub(r'sql\s*\n', '', sql_query, flags=re.IGNORECASE).strip()
        
        # Remove leading/trailing quotes and extra whitespace
        sql_query = sql_query.strip('\'"').strip()
        
        # Remove explanatory text and extract SQL
        lines = sql_query.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            # Keep SQL keywords and query parts
            if (line.upper().startswith(('SELECT', 'WITH', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'UNION')) or
                line.upper().startswith(('AND', 'OR', ')', '(')) or
                (sql_lines and not line.upper().startswith(('THE', 'THIS', 'HERE', 'QUERY', 'SQL', 'GENERATED')))):
                sql_lines.append(line)
        
        sql_query = '\n'.join(sql_lines).strip()
        
        # If still too short, try to extract SELECT statement
        if len(sql_query) < 10:
            select_match = re.search(r'(SELECT.*?(?:LIMIT\s+\d+|$))', response_text, re.IGNORECASE | re.DOTALL)
            if select_match:
                sql_query = select_match.group(1).strip()
        
        return sql_query

    def get_sample_queries_from_schema(self, schema_data: List[Dict[str, Any]]) -> List[str]:
        """Generate intelligent sample queries based on actual database schema."""
        if not schema_data:
            return [
                "Show me all data",
                "Count total records",
                "What is the latest data?"
            ]
        
        samples = []
        
        for table in schema_data[:3]:  # Limit to first 3 tables
            table_name = table.get('table_name', '')
            if not table_name:
                continue
                
            columns = table.get('columns', [])
            
            # Basic queries
            samples.extend([
                f"Show me all {table_name}",
                f"Count how many {table_name} we have",
                f"Show me the first 10 {table_name}"
            ])
            
            # Column-specific queries
            for column in columns[:2]:  # First 2 interesting columns
                col_name = column.get('column_name', '')
                col_type = column.get('data_type', '')
                
                if col_name in ['id', 'created_at', 'updated_at']:
                    continue
                    
                if 'amount' in col_name.lower() or 'price' in col_name.lower() or 'value' in col_name.lower():
                    samples.append(f"What's the total {col_name} in {table_name}?")
                    samples.append(f"Show me {table_name} with highest {col_name}")
                elif 'name' in col_name.lower() or col_type in ['text', 'varchar']:
                    samples.append(f"Find {table_name} by {col_name}")
                elif 'date' in col_name.lower() or 'time' in col_name.lower():
                    samples.append(f"Show me recent {table_name} by {col_name}")
        
        # Add some general analytical queries
        if len(schema_data) > 1:
            samples.extend([
                "Show me data relationships between tables",
                "What are the most common values?",
                "Give me a summary of all tables"
            ])
        
        return samples[:8]  # Return max 8 samples

    def explain_query_results(self, natural_query: str, sql_query: str,results: List[Dict[str, Any]], column_info: Dict[str, str]) -> str:
        """Enhanced result explanation with better insights."""
        try:
            if not results:
                return "No results found for your query. Try adjusting your search criteria or check the available data."
            
            result_count = len(results)
            
            # Handle single-value results (aggregates)
            if result_count == 1 and len(results[0]) == 1:
                key = list(results[0].keys())[0]
                value = results[0][key]
                
                if value is None:
                    return "The query returned no value (NULL). This might indicate no matching data was found."
                
                # Enhanced aggregate explanations
                if 'count' in key.lower() or 'COUNT(' in sql_query.upper():
                    entity = self._extract_entity_from_query(natural_query, sql_query)
                    return f"**Total Count**: {value:,} {entity} found in your database."
                
                if 'sum' in key.lower() or 'SUM(' in sql_query.upper():
                    return f"**Total Sum**: {value:,} (combined total from all matching records)"
                
                if 'avg' in key.lower() or 'AVG(' in sql_query.upper():
                    return f"**Average Value**: {value:,.2f} (mean across all records)"
                
                if 'max' in key.lower() or 'MAX(' in sql_query.upper():
                    return f"**Maximum Value**: {value:,} (highest value found)"
                
                if 'min' in key.lower() or 'MIN(' in sql_query.upper():
                    return f"**Minimum Value**: {value:,} (lowest value found)"
                
                # Generic single value
                label = column_info.get(key, key.replace("_", " ").title())
                return f" **{label}**: {value:,}" if isinstance(value, (int, float)) else f" **{label}**: {value}"
            
            # Handle multiple rows
            if result_count <= 10:
                return self._format_small_result_explanation(results, natural_query, result_count)
            else:
                return self._format_large_result_explanation(results, natural_query, result_count)
        
        except Exception as e:
            logger.error(f"Error in explain_query_results: {e}")
            return f" Found {len(results)} result(s). Results are displayed in the table below."

    def _extract_entity_from_query(self, natural_query: str, sql_query: str) -> str:
        """Extract the entity being counted from the query."""
        # Try to identify from natural language first
        for entity in ['customers', 'orders', 'products', 'users', 'payments', 'transactions']:
            if entity in natural_query.lower():
                return entity
        
        # Try to extract from SQL query
        from_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
        if from_match:
            table_name = from_match.group(1)
            return table_name.lower()
        
        return "records"

    def _format_small_result_explanation(self, results: List[Dict[str, Any]], natural_query: str, count: int) -> str:
        """Format explanation for small result sets."""
        explanation = f" **Found {count} result{'s' if count != 1 else ''}** matching your query."
        
        if count == 1:
            # Single result - show key details
            row = results[0]
            key_fields = []
            for key, value in list(row.items())[:3]:  # Show first 3 fields
                clean_key = key.replace("_", " ").title()
                if isinstance(value, (int, float)):
                    key_fields.append(f"{clean_key}: {value:,}")
                else:
                    key_fields.append(f"{clean_key}: {value}")
            
            if key_fields:
                explanation += f"\n\n **Key Details**: {' | '.join(key_fields)}"
        
        else:
            # Multiple results - show summary
            columns = list(results[0].keys())
            explanation += f" The results contain {len(columns)} column{'s' if len(columns) != 1 else ''}: {', '.join([col.replace('_', ' ').title() for col in columns[:4]])}{'...' if len(columns) > 4 else ''}."
        
        return explanation

    def _format_large_result_explanation(self, results: List[Dict[str, Any]], natural_query: str, count: int) -> str:
        """Format explanation for large result sets."""
        columns = list(results[0].keys())
        
        explanation = f" **Large Dataset Found**: {count:,} records with {len(columns)} columns."
        
        # Identify key numeric columns for insights
        numeric_insights = []
        for col in columns[:3]:  # Check first 3 columns
            try:
                values = [row[col] for row in results[:100] if row[col] is not None]  # Sample first 100
                if values and all(isinstance(v, (int, float)) for v in values):
                    avg_val = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    numeric_insights.append(f"{col.replace('_', ' ').title()}: avg {avg_val:.1f}, range {min_val:,}-{max_val:,}")
            except:
                continue
        
        if numeric_insights:
            explanation += f"\n\n **Quick Insights**: {' | '.join(numeric_insights[:2])}"
        
        explanation += f"\n\n **Tip**: Results are limited to first 1000 rows for performance. Use filters to narrow down your search."
        
        return explanation


    def suggest_visualization_options(self, results: List[Dict[str, Any]], natural_query: str) -> List[Dict[str, Any]]:
        """Enhanced visualization suggestions with detailed recommendations and SMART COLUMN HINTS."""
        if not results:
            return []

        row_count = len(results)
        columns = list(results[0].keys()) if results else []
        
        if not columns:
            return []

        # Analyze column types - IMPROVED DETECTION
        numeric_columns = []
        text_columns = []
        date_columns = []
        potential_date_columns = [] # For columns that might be dates but need parsing

        for col in columns:
            sample_values = [row[col] for row in results[:20] if row[col] is not None]
            if not sample_values:
                continue
            
            # Check if numeric
            if all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit()) for v in sample_values):
                numeric_columns.append(col)
            # Check if date/time related (more robust)
            elif any(keyword in str(sample_values[0]).lower() for keyword in ['date', 'time', 'year', 'month', 'day']) or \
                 any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                # Further check if it looks like a date string
                try:
                    # Try parsing a few samples
                    parsed_dates = pd.to_datetime(sample_values[:5], errors='raise')
                    date_columns.append(col)
                except (ValueError, TypeError):
                    # If parsing fails, still consider it a potential date column for user selection
                    potential_date_columns.append(col)
            else:
                text_columns.append(col)

        # Combine date columns (confirmed) and potential date columns for chart axes
        all_date_like_columns = date_columns + potential_date_columns

        suggestions = []

        # Always suggest table view first
        suggestions.append({
            'type': 'table',
            'title': 'Data Table',
            'description': f'Complete tabular view of all {row_count:,} records',
            'priority': 1,
            'suitable_for': 'Viewing all data details and performing analysis'
            # No specific column hints needed for table
        })

        # Bar chart for categorical data - SMART HINTS
        if text_columns and numeric_columns and row_count <= 100: # Increased row limit slightly
            # Heuristic: Prefer shorter text columns with fewer unique values for X-axis
            candidate_x_cols = sorted(text_columns, key=lambda c: (len(set(str(row[c]) for row in results))), reverse=False)
            best_x_col = candidate_x_cols[0] if candidate_x_cols else text_columns[0]
            best_y_col = numeric_columns[0] # Usually the first numeric col is the measure

            unique_categories = len(set(str(row[best_x_col]) for row in results))
            if unique_categories <= 25: # Increased category limit slightly
                suggestions.append({
                    'type': 'bar',
                    'title': f'{best_y_col} by {best_x_col}',
                    'description': f'Compare {best_y_col} across {best_x_col} categories',
                    'priority': 2,
                    'suitable_for': f'Comparing values across {unique_categories} categories',
                    # --- SMART COLUMN HINTS ---
                    'x': best_x_col,
                    'y': best_y_col
                })

        # Line chart for time series - SMART HINTS
        if all_date_like_columns and numeric_columns:
            best_x_col = all_date_like_columns[0] # Assume first date-like is the time dimension
            best_y_col = numeric_columns[0] # Usually the first numeric col is the measure

            suggestions.append({
                'type': 'line',
                'title': f'{best_y_col} over {best_x_col}',
                'description': f'Show trends of {best_y_col} over {best_x_col}',
                'priority': 2,
                'suitable_for': 'Analyzing trends and patterns over time',
                # --- SMART COLUMN HINTS ---
                'x': best_x_col,
                'y': best_y_col
            })

        # Scatter plot for correlation - SMART HINTS
        if len(numeric_columns) >= 2:
            x_col = numeric_columns[0]
            y_col = numeric_columns[1] # Use the second numeric column

            suggestions.append({
                'type': 'scatter',
                'title': f'{y_col} vs {x_col}',
                'description': f'Explore relationship between {x_col} and {y_col}',
                'priority': 3,
                'suitable_for': 'Finding correlations between numeric variables',
                # --- SMART COLUMN HINTS ---
                'x': x_col,
                'y': y_col,
                'size': "None", # Default to no size encoding
                'color': "None" # Default to no color encoding
            })

        # Pie chart for distributions - SMART HINTS
        if text_columns and numeric_columns and row_count <= 20: # Keep row limit tight
            # Heuristic: Prefer text columns with fewer unique values for labels
            candidate_label_cols = sorted(text_columns, key=lambda c: len(set(str(row[c]) for row in results)))
            best_labels_col = candidate_label_cols[0] if candidate_label_cols else text_columns[0]
            best_values_col = numeric_columns[0] # Usually the first numeric col is the measure

            unique_categories = len(set(str(row[best_labels_col]) for row in results))
            if unique_categories <= 10: # Keep category limit tight for pie charts
                suggestions.append({
                    'type': 'pie',
                    'title': f'Distribution of {best_values_col} by {best_labels_col}',
                    'description': f'Show distribution of {best_values_col} by {best_labels_col}',
                    'priority': 3,
                    'suitable_for': f'Showing proportional breakdown of {unique_categories} categories',
                    # --- SMART COLUMN HINTS ---
                    'labels': best_labels_col,
                    'values': best_values_col
                })

        # Histogram for distribution analysis - SMART HINTS
        if numeric_columns and row_count > 10: # Lowered row limit for histogram relevance
            best_col = numeric_columns[0] # Usually the first numeric col

            suggestions.append({
                'type': 'histogram',
                'title': f'Distribution of {best_col}',
                'description': f'Show distribution pattern of {best_col}',
                'priority': 4,
                'suitable_for': 'Understanding data distribution and frequency patterns',
                # --- SMART COLUMN HINTS (for histogram, x is the column to analyze) ---
                'x': best_col # This tells the visualizer which column to put on the x-axis
            })

        # Sort by priority and return top suggestions
        suggestions.sort(key=lambda x: x['priority'])
        return suggestions[:5]  # Return top 5 suggestions

    def get_connection_test_query(self) -> str:
        """Get a simple query to test database connection."""
        # return "SELECT 1 as connection_test, CURRENT_TIMESTAMP as current_time;", '', query, flags=re.MULTILINE)
        return "SELECT 1 as connection_test, CURRENT_TIMESTAMP as current_time;"
