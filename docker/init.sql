CREATE TABLE customers (
  id integer PRIMARY KEY,
  region text NOT NULL,
  segment integer NOT NULL
);
CREATE TABLE orders (
  id bigint PRIMARY KEY,
  customer_id integer NOT NULL REFERENCES customers(id),
  amount numeric(12,2) NOT NULL,
  status integer NOT NULL
);
INSERT INTO customers
SELECT i, (ARRAY['north','south','east','west'])[1 + (i % 4)], i % 17
FROM generate_series(1, 12000) AS g(i);
INSERT INTO orders
SELECT i, 1 + (i * 37 % 12000), ((i * 13) % 10000) / 10.0, i % 5
FROM generate_series(1, 120000) AS g(i);
CREATE INDEX orders_customer_id_idx ON orders(customer_id);
CREATE INDEX customers_region_idx ON customers(region);
ANALYZE;
