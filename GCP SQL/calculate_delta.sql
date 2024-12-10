CREATE OR REPLACE FUNCTION calculate_delta()
RETURNS trigger 
LANGUAGE plpgsql
AS $$ 
BEGIN 
-- Calculate the delta for the current row 
NEW.delta := NEW.bikes - ( SELECT bikes FROM velib_status WHERE station = NEW.station AND id < NEW.id ORDER BY id DESC LIMIT 1 );
RETURN NEW; 
END; 
$$
