import psycopg2
from Website import Website

class PostgresDatabase:
    def __init__(self) -> None:
        self.connection = None
        self.cursor = None

    def connect(self):
        """! Connect to the postgres database. To view the database or debug open up the shell for the 
        database image then use the 'psql -U docker maindb' command to enter into the database bash. Then
        use the '\l' command to list the databases.
        """
        try:
            conn = psycopg2.connect(database="postgres", user="postgres", password="postgres", host="localhost")
            self.connection = conn
        except:
            pass

        # Create quotes table if none exists
        self.execute("""
        CREATE TABLE IF NOT EXISTS site_data (
            id serial PRIMARY KEY not null, 
            name varchar not null,
            initial_url varchar not null, 
            visited_url varchar not null, 
            text varchar not null,
            category varchar not null,
            tags varchar not null,
            timestamp timestamp default current_timestamp
            );
        """)

    def close(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def commit(self) -> None:
        # assert self.connection, 'Database connection is not established'
        if self.connection:
            self.connection.commit()

    def execute(self, command, values=None) -> None:
        # assert self.connection, 'Database connection is not established'
        if self.connection:
            self.cursor = self.connection.cursor()
            self.cursor.execute(command, values)

    def add_website(self, site: Website) -> None:
        
        if site.i_url == "missing note" or site.v_url == "" or site.text == "":
            return
        
        # Check to see if text is already in database
        self.cursor.execute(
            "SELECT * FROM site_data WHERE initial_url = %s;", (site.i_url,))
        result = self.cursor.fetchone()

        # If it is in DB, create log message
        if result:
            print(f"Item already in database: {site.name}")
        else:
            # Define insert statement
            self.cursor.execute("""INSERT INTO site_data (name, initial_url, visited_url, text, category, tags) VALUES (%s,%s,%s,%s,%s,%s);""", (
                site.name,
                site.i_url,
                site.v_url,
                site.text,
                site.category,
                site.tags,
            ))

        # Execute insert of data into database
        self.commit()