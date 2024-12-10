CREATE TRIGGER trigger_calculate_delta BEFORE INSERT OR UPDATE ON public.velib_status FOR EACH ROW EXECUTE FUNCTION calculate_delta();
