from threading import Timer

def debounce(wait): 
    def decorator(fun):
        def debounced(*args, **kwargs):
            def call_it():
                fun(*args, **kwargs)
 
            try:
                debounced.t.cancel()
            except AttributeError:
                pass
 
            debounced.t = Timer(wait, call_it)
            debounced.t.start()
 
        return debounced
 
    return decorator