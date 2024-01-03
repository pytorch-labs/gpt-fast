import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.profiler import profile, record_function, ProfilerActivity

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.joint_graph_constant_folding = False # avoid OOM bug for DCFormer-6.9B
#torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

#from model import Transformer
from model_talking_train import Transformer, ModelArgs, apply_activation_checkpointing
from tp import maybe_init_dist


def train(mod, data, opt):
    #opt.zero_grad(True)
    pred = mod(data[0]) # BTV
    loss = torch.nn.CrossEntropyLoss()(pred.view((-1,pred.shape[-1])), data[1].view((-1)))
    loss.backward()
    #opt.step()

def timed(fn):
    #t0 = time.time()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000
    #return result, time.time()-t0 

########################## Llama 7B
# no window and query_chunk
# llama L=32 no compile 0.7129 
# llama L=32 compile 0.5486 (no checkpointing), 0.6863(after checkpointing) 1.25
# llama L=20 compile 0.34966(no checkpointing), 0.4387(after checkpointing) 1.25
# llama L=10 compile 0.19117(no checkpointing), 0.2354(after checkpointing) 1.23
# llama L=2 compile med 0.06585(forward+backward) 


# with window and query_chunk=1024
# llama L=32 compile 0.6565(after checkpointing), 0.5253(no checkpointing) 
# llama talking L=32 compile 1.4086(2.14, after checkpointing) 

# with window and query_chunk=512
# llama L=32 compile 0.64864(4min, after checkpointing), 0.5212(no checkpointing)
# llama talking L=32 compile 1.130 (1.74, 19.5min, after checkpointing) 

# with window and query_chunk=256
# llama L=32 compile 0.6535(6.3min, after checkpointing), 0.5249(no checkpointing) 
# llama talking L=32 compile 1.041(1.59, 33min, after checkpointing)  # 0.8047(29.15min, no checkpointing) 

# with window and query_chunk=128
# llama L=32 compile 0.6903(9min, after checkpointing) # no constant_folding
# llama talking L=32 compile 0.9604(1.39(with same query_chunk),1.48(best vs best), 60min, after checkpointing) # no constant_folding  
# llama L=20 compile 0.4329(5min, after checkpointing) 
# llama L=10 compile 0.2308(4min, after checkpointing) 

# with window and query_chunk=64
# llama talking L=32 compile 0.8021(118min, no checkpointing) # no constant_folding  

# with window and query_chunk_size=128
# llama talking no compile L=32 2.879 
# llama talking compile L=32 0.9604(60min, after checkpointing) # no constant_folding  
# llama talking compile L=32 0.760(1.38(query_chunk_size=128 vs 2048), 1.46(best vs best) 56min, no checkpointing)# no constant_folding   
# llama talking compile L=20 0.479039(1.37, 37min)  
# llama talking compile L=10 0.2537(1.32, 14.5min)  
# llama talking compile L=2  0.0785(1.19) 


############################# Llama13B
# llama compile L=40   0.9324 (4.5min, no checkpointing, with window and query_chunk_size=512)
# llama talking compile L=40  (min, no checkpointing, with window and query_chunk_size=128, no constant_folding)   

def main():
    N_ITERS = 100
    VS = 50257
    device = torch.device('cuda:0')
    max_seq_length = 2048 // 1
    dtype = torch.float16
    B, T, S, N, D, I = 1, 2048 // 1, 2048 // 1, 32, 128, 2; E = D * N; #L=32 // 8 # 6.7B
    #config = ModelArgs(n_layer=32,n_head=32,dim=4096,block_size=max_seq_length) # 6.7B
    config = ModelArgs(n_layer=40,n_head=40,dim=5120,block_size=max_seq_length) # 13B
    #config.n_layer = 2 
    model = Transformer(config)
    model = model.to(device=device, dtype=dtype)
    model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
    for layer in model.layers:
        if hasattr(layer.attention, 'dyn_w_proj'):
            layer.attention.dyn_w_proj.merge_weights()

    #opt = torch.optim.Adam(model.parameters())
    opt = None
    #“default”, “reduce-overhead”, “max-autotune” or “max-autotune-no-cudagraphs”
    #train_opt = torch.compile(train, mode="reduce-overhead",fullgraph=False)
    #train_opt = torch.compile(train, mode="default",fullgraph=False, dynamic=False)
    train_opt = torch.compile(train, mode="default",fullgraph=False)
    #model = torch.compile(model)
    #model = apply_activation_checkpointing(model)

    print('start training')
    compile_times = []
    inp = torch.randint(VS, (B,T+1)).to(device)
    inp = [inp[:,:-1], inp[:, 1:]]
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    if True:
        for i in range(N_ITERS):
            if i == 50:
                t0=time.time()
            _, compile_time = timed(lambda: train_opt(model, inp, opt))
            #_, compile_time = timed(lambda: train(model, inp, opt))
            compile_times.append(compile_time)
            print(f"compile train time {i}: {compile_time}")
        #torch.cuda.synchronize()
        print('time for 50 iterations:', time.time()-t0)
        print("~" * 10)
    
    #prof.export_chrome_trace("trace_llama_l2_querychunk128_talking_no-checkpointing_compiled_train.json")
    #eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    
    #speedup = eager_med / compile_med
    #assert(speedup > 1)
    #print(f"(train) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print('compile med', compile_med, np.array(compile_times)[50:].mean())
    #print("~" * 10)






if  __name__ == '__main__':
    main()
