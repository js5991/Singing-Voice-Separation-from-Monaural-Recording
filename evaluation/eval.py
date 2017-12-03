from bss_eval import *

def eval_result(s, s_estimate, m):
    sdr_voice, sir_voice, sar_voice, perm_voice = bss_eval_sources(s, s_estimate)
    sdr, sir, sar, perm = bss_eval_sources(s, m)
    #nsdr = sdr_voice - sdr
    #return nsdr
    return sdr_voice, sir_voice, sar_voice, sdr, sir, sar