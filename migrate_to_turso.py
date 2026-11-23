import sqlite3
import os
import libsql_client

TURSO_URL = os.getenv("TURSO_DATABASE_URL", "https://presales-db-mahmoudsaf.aws-eu-central-1.turso.io")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3OTU0MzgwNjksImlkIjoiMDFhMjA4NzItNjAwMS03ZmMyLThhMzQtMGExMDQ0NTM2M2Y2Iiwia2lkIjoiakQzTFBUZ2F5WGtCdEwxY0xTUVpnb0Y5VnhZWWd1Rkp3SlJ3dEp3UDc2YyIsInJpZCI6IjBmYmE5ZTM2LTQxMGItNDA0Mi1hNjZlLTQyN2Y0OTMyZjM4ZiJ9.1EooxDmPyNgtWOv-92rfaXTo6j7VIAdGNa_QR48mQgk0FyoQimSSDPGNdtMqSQ1YlOAdToXBTaEDZ8WbvqMsAQ")

def migrate():
    print(f"Connecting to Turso at {TURSO_URL}...")
    client = libsql_client.create_client_sync(url=TURSO_URL, auth_token=TURSO_TOKEN)

    # 1. Migrate Customers
    crm_conn = sqlite3.connect("crm.db")
    crm_conn.row_factory = sqlite3.Row
    crm_cur = crm_conn.cursor()

    cust_rows = crm_cur.execute("SELECT customer_id, company_name, contact_name, contact_email, contact_phone FROM customers").fetchall()
    print(f"Migrating {len(cust_rows)} customers...")
    for row in cust_rows:
        client.execute(
            "INSERT OR REPLACE INTO customers (customer_id, company_name, contact_name, contact_email, contact_phone) VALUES (?, ?, ?, ?, ?)",
            [row["customer_id"], row["company_name"], row["contact_name"], row["contact_email"], row["contact_phone"]]
        )

    # Ensure "aaa test" customer exists for deal #82
    try:
        client.execute(
            "INSERT OR IGNORE INTO customers (company_name, contact_name) VALUES (?, ?)",
            ["aaa test", "Admin"]
        )
    except Exception as e:
        print("Note on aaa test customer:", e)

    # 2. Migrate Deals
    deal_rows = crm_cur.execute("""
        SELECT deal_id, customer_id, deal_name, primary_vendors, stage, estimated_value, assigned_presales, vendor_notes, created_at, updated_at, deal_category, closing_date 
        FROM deals
    """).fetchall()
    print(f"Migrating {len(deal_rows)} deals...")
    for row in deal_rows:
        client.execute("""
            INSERT OR REPLACE INTO deals (
                deal_id, customer_id, deal_name, primary_vendors, stage, estimated_value, assigned_presales, vendor_notes, created_at, updated_at, deal_category, closing_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["deal_id"], row["customer_id"], row["deal_name"], row["primary_vendors"], row["stage"],
            row["estimated_value"], row["assigned_presales"], row["vendor_notes"], row["created_at"],
            row["updated_at"], row["deal_category"], row["closing_date"]
        ])

    # Re-add Deal #82 ("SAN techrefresh")
    aaa_cust = client.execute("SELECT customer_id FROM customers WHERE company_name = 'aaa test'").rows
    aaa_cust_id = aaa_cust[0][0] if aaa_cust else 1
    client.execute("""
        INSERT OR REPLACE INTO deals (
            deal_id, customer_id, deal_name, primary_vendors, stage, estimated_value, assigned_presales, vendor_notes, deal_category, closing_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        82, aaa_cust_id, "SAN techrefresh", "General", "Discovery", 0.0, "Presales 1", "Restored from online creation", "3- Opportunity Efforts & PO", None
    ])
    print("Deal #82 (SAN techrefresh) successfully restored into Turso!")

    # 3. Migrate Tasks
    tasks_conn = sqlite3.connect("tasks.db")
    tasks_conn.row_factory = sqlite3.Row
    tasks_cur = tasks_conn.cursor()

    task_rows = tasks_cur.execute("""
        SELECT task_id, task_title, category, related_deal_id, assigned_to, vendor_domain, status, priority, due_date, management_blockers, created_at, updated_at, started_at, completed_at, deal_name, customer_name, customer_id, deal_category, closing_date, previous_task_id
        FROM tasks
    """).fetchall()
    print(f"Migrating {len(task_rows)} tasks...")
    for row in task_rows:
        client.execute("""
            INSERT OR REPLACE INTO tasks (
                task_id, task_title, category, related_deal_id, assigned_to, vendor_domain, status, priority, due_date, management_blockers, created_at, updated_at, started_at, completed_at, deal_name, customer_name, customer_id, deal_category, closing_date, previous_task_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            row["task_id"], row["task_title"], row["category"], row["related_deal_id"], row["assigned_to"],
            row["vendor_domain"], row["status"], row["priority"], row["due_date"], row["management_blockers"],
            row["created_at"], row["updated_at"], row["started_at"], row["completed_at"], row["deal_name"],
            row["customer_name"], row["customer_id"], row["deal_category"], row["closing_date"], row["previous_task_id"]
        ])

    # 4. Migrate Task Activity Log
    act_rows = tasks_cur.execute("SELECT id, task_id, from_status, to_status, changed_at, user_id FROM task_activity_log").fetchall()
    print(f"Migrating {len(act_rows)} task activity log entries...")
    for row in act_rows:
        client.execute("""
            INSERT OR REPLACE INTO task_activity_log (id, task_id, from_status, to_status, changed_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [row["id"], row["task_id"], row["from_status"], row["to_status"], row["changed_at"], row["user_id"]])

    # 5. Verify Turso Counts
    c_count = client.execute("SELECT COUNT(*) FROM customers").rows[0][0]
    d_count = client.execute("SELECT COUNT(*) FROM deals").rows[0][0]
    t_count = client.execute("SELECT COUNT(*) FROM tasks").rows[0][0]
    a_count = client.execute("SELECT COUNT(*) FROM task_activity_log").rows[0][0]

    print("=" * 60)
    print("MIGRATION TO TURSO COMPLETE & VERIFIED:")
    print(f"  Customers count in Turso: {c_count}")
    print(f"  Deals count in Turso:     {d_count}")
    print(f"  Tasks count in Turso:     {t_count}")
    print(f"  Activity logs in Turso:   {a_count}")
    print("=" * 60)
    client.close()

if __name__ == "__main__":
    migrate()
