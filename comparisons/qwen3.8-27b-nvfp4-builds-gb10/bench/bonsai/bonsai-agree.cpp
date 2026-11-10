#include "llama.h"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
int main(int argc,char**argv){
 if(argc!=4)return 2;
 json answers;std::ifstream(argv[2])>>answers;
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=999;
 auto*model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto*vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);
 auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;cp.n_threads=4;cp.n_threads_batch=4;cp.type_k=GGML_TYPE_BF16;cp.type_v=GGML_TYPE_BF16;cp.no_perf=true;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
 auto*ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 auto batch=llama_batch_init(512,0,1);json records=json::array();long total=0,agree=0;std::map<std::string,std::pair<long,long>>cats;
 for(auto&r:answers["records"]){
  auto prompt=r["prompt_ids"].get<std::vector<int>>();auto ans=r["answer_ids"].get<std::vector<int>>();auto tokens=prompt;tokens.insert(tokens.end(),ans.begin(),ans.end());
  if(tokens.size()>4096)throw std::runtime_error("context exceeds reference 4096");
  llama_memory_clear(llama_get_memory(ctx),true);json predicted=json::array();long ok=0;
  for(size_t off=0;off<tokens.size()-1;off+=512){
   batch.n_tokens=std::min<size_t>(512,tokens.size()-1-off);
   for(int i=0;i<batch.n_tokens;++i){size_t pos=off+i;batch.token[i]=tokens[pos];batch.pos[i]=pos;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=pos+1>=prompt.size();}
   if(llama_decode(ctx,batch))throw std::runtime_error("decode failed");
   for(int i=0;i<batch.n_tokens;++i){size_t pos=off+i;if(pos+1<prompt.size())continue;auto*logits=llama_get_logits_ith(ctx,i);if(!logits)throw std::runtime_error("missing logits");int top=std::max_element(logits,logits+nv)-logits;predicted.push_back(top);ok+=top==tokens[pos+1];}
  }
  if(predicted.size()!=ans.size())throw std::runtime_error("position accounting mismatch");
  std::string cat=r["category"];cats[cat].first+=ok;cats[cat].second+=ans.size();total+=ans.size();agree+=ok;
  records.push_back({{"id",r["id"]},{"category",cat},{"n",ans.size()},{"agree",ok},{"argmax_ids",predicted}});
  json by=json::object();for(auto&[c,n]:cats)by[c]=double(n.first)/n.second;
  json out={{"name","Bonsai 2 Q2_0 llama.cpp b12242"},{"method","Teacher-forced BF16 prompt_ids + answer_ids; unsampled raw-logit argmax at every answer position, BF16 K/V, fresh recurrent/KV state per record, batch/ubatch 512, no draft or prefix reuse"},{"summary",{{"positions",total},{"agree",agree},{"agreement",double(agree)/total},{"by_category",by}}},{"records",records}};
  std::ofstream(argv[3])<<out.dump(1)<<'\n';std::cout<<r["id"]<<" "<<ok<<"/"<<ans.size()<<std::endl;
 }
 llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return total==46706?0:5;
}
