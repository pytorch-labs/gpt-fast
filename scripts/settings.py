# Create a counter for failed cases to compute acceptance rate
def init_counter():
    global counter
    counter = 0

def init_texts():
    global texts
    texts  = []

# n is accepted tokens per call to target model 
def init_counter_n_list():
    global counter_n_list
    counter_n_list = []