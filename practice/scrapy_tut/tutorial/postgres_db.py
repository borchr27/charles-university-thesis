import psycopg2
from website import Website

class PostgresDatabase:
    def __init__(self):
        # Connection Details
        hostname = 'localhost'
        username = 'postgres'
        password = 'postgres'
        database = 'postgres'

        # Create/Connect to database
        self.connection = psycopg2.connect(
            host=hostname, user=username, password=password, dbname=database)

        # Create cursor, used to execute commands
        self.cur = self.connection.cursor()

        # Create quotes table if none exists
        self.cur.execute("""
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

    def add_item(self, site: Website):

        # Check to see if text is already in database
        self.cur.execute(
            "SELECT * FROM site_data WHERE initial_url = %s", (site.i_url,))
        result = self.cur.fetchone()

        # If it is in DB, create log message
        if result:
            print(f"Item already in database: {site.name}")
        else:
            # Define insert statement
            self.cur.execute("""INSERT INTO site_data (name, initial_url, visited_url, text, category, tags) VALUES (%s,%s,%s,%s,%s,%s)""", (
                site.name,
                site.i_url,
                site.v_url,
                site.text,
                site.category,
                site.tags,
            ))

        # Execute insert of data into database
        self.connection.commit()

    def db_close(self):
        # Close cursor & connection to database
        self.cur.close()
        self.connection.close()
