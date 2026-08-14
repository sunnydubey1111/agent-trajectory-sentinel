-- Fixture for SQLDatabaseTool: a tiny read-only e-commerce database.
--
-- This is a TOOL FIXTURE, not research data. It used to live as a committed
-- binary at traces/ecommerce.db, which put a fixture inside the frozen episode
-- corpus, shipped it to the published Hugging Face dataset, and left it
-- unhashed by BASELINE_MANIFEST.json with nothing able to regenerate it.
-- Here it is plain SQL: diffable, reviewable, and rebuilt on demand into the
-- runtime directory by `SQLDatabaseTool._ensure_db`.
--
-- Row values are unchanged from the original binary; edit this file to change
-- the fixture, then delete the built copy under runs/fixtures/.

CREATE TABLE orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        quantity INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products (id)
    );
INSERT INTO "orders" VALUES(1,1,2,'2026-07-10 14:32:00','Delivered');
INSERT INTO "orders" VALUES(2,3,1,'2026-07-11 09:15:00','Delivered');
INSERT INTO "orders" VALUES(3,2,1,'2026-07-12 11:45:00','Shipped');
INSERT INTO "orders" VALUES(4,5,3,'2026-07-13 16:20:00','Processing');
INSERT INTO "orders" VALUES(5,7,2,'2026-07-14 08:30:00','Processing');
INSERT INTO "orders" VALUES(6,1,1,'2026-07-14 10:12:00','Cancelled');
CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER NOT NULL,
        description TEXT
    );
INSERT INTO "products" VALUES(1,'Wireless Noise-Canceling Headphones','Electronics',129.99,45,'Premium over-ear Bluetooth headphones with active noise cancellation.');
INSERT INTO "products" VALUES(2,'Ergonomic Office Chair','Furniture',189.5,15,'High-back mesh chair with adjustable lumbar support and armrests.');
INSERT INTO "products" VALUES(3,'Stainless Steel Water Bottle','Home & Kitchen',24.99,120,'Double-walled vacuum insulated bottle, keeps drinks cold for 24 hours.');
INSERT INTO "products" VALUES(4,'Mechanical Gaming Keyboard','Electronics',79.99,30,'Tactile blue switch mechanical keyboard with RGB backlighting.');
INSERT INTO "products" VALUES(5,'Organic Matcha Green Tea Powder','Grocery',18.75,80,'Ceremonial grade pure stone-ground Japanese green tea powder.');
INSERT INTO "products" VALUES(6,'Smart Fitness Tracker','Electronics',49.99,65,'Waterproof fitness band with heart rate monitor and sleep tracking.');
INSERT INTO "products" VALUES(7,'Ceramic Coffee Mug Set','Home & Kitchen',32.0,25,'Set of 4 hand-painted ceramic mugs, microwave and dishwasher safe.');
INSERT INTO "products" VALUES(8,'Running Shoes','Apparel',85.0,50,'Lightweight breathable athletic sneakers with cushioned sole.');
INSERT INTO "products" VALUES(9,'Cast Iron Skillet (10-inch)','Home & Kitchen',29.99,40,'Pre-seasoned cast iron skillet for stove-top, oven, or campfire cooking.');
INSERT INTO "products" VALUES(10,'Adjustable Dumbbells Set','Sports',249.99,10,'Compact adjustable weight dial dumbbells, pair up to 52.5 lbs each.');
