import sqlite3
import akshare_tools

DATABASE_FILE = './db/stock_list.db'


def normalize_stock_code(code):
    """将无前缀股票代码转换为带前缀的代码(sh/ sz)。"""
    if not code:
        return None
    code = str(code).strip()
    if code.startswith(('sh', 'sz')):
        return code
    if code.startswith('6'):
        return f"sh{code}"
    return f"sz{code}"


def query_by_code(code):
    normalized_code = normalize_stock_code(code)
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT code, name FROM stock_catalog WHERE code = ?', (normalized_code,))
    row = cursor.fetchone()
    conn.close()
    return row


def query_by_name_keyword(keyword):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # 使用 LIKE 进行模糊匹配（% 表示任意字符）
    cursor.execute('SELECT code, name FROM stock_catalog WHERE name LIKE ?', (f'%{keyword}%',))
    rows = cursor.fetchall()
    conn.close()
    return rows


def fetch_and_store_stock_list():
    """获取股票列表并存储到数据库"""
    stock_list = akshare_tools.get_stock_list()
    if stock_list is not None:
        # 连接数据库
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # 3. 清空旧数据（可选，避免重复）
        cursor.execute('DELETE FROM stock_catalog')
        
        # 插入数据到表中
        for _, row in stock_list.iterrows():
            code = normalize_stock_code(row['code'])
            name = row['name']
            try:
                cursor.execute('INSERT OR IGNORE INTO stock_catalog (code, name) VALUES (?, ?)', (code, name))
            except Exception as e:
                print(f"插入股票 {code} - {name} 时出错: {e}")

        conn.commit()
        conn.close()
        print("股票列表已成功存储到数据库！")
    else:
        print("未能获取股票列表，无法存储到数据库。")


if __name__ == "__main__":

    # fetch_and_store_stock_list()


    result = query_by_name_keyword("万科")
    print("查询结果:", result)