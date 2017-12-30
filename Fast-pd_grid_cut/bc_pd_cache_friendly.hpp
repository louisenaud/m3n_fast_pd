//
//  bc_pd_cache_friendly.hpp
//  All_Fast_PD
//
//  Created by Bruno CONEJO on 1/7/17.
//  Copyright © 2017 Bruno CONEJO. All rights reserved.
//

#ifndef bc_pd_cache_friendly_hpp
#define bc_pd_cache_friendly_hpp
#include <vector>
#include <cassert>
#include <time.h>

using namespace std;

typedef int               mrf_data;
typedef int               mrf_label;
typedef int               mrf_ind;
typedef double            mrf_nrg;

typedef vector<mrf_label> mrf_v_label;
typedef vector<mrf_ind>   mrf_v_ind;
typedef vector<mrf_data>  mrf_v_data;


#include "GridGraph_2D_4C.h"
typedef GridGraph_2D_4C<mrf_data, mrf_data, mrf_nrg> GraphType4C;

//#include "GridGraph_2D_4C_MT.h"
//typedef GridGraph_2D_4C_MT<mrf_data, mrf_data, mrf_nrg> GraphType4C_MT;

class BC_PD_CF{
  
  
public:
  BC_PD_CF(int R, int C, int L, mrf_data *unaries, mrf_data *weights,
           mrf_data *dist, mrf_data *x_init);
  
  ~BC_PD_CF();
  
  void optimize(const size_t I_max, const bool grow_sink);
  void restore_unaries();
  
  mrf_nrg get_primal_nrg(){return nrg;};
  mrf_v_label get_solution();
  
  mrf_label get_solution(int i, int j) {return x(node(i, j));};
  void get_dual_variables(double *h_out, double *yh_out, double *yv_out);
  
private:
  // Some size
  int N, L, E, R, C;
  int RE, CE, NE, YOFS, RB;
  
  // Pointers to mrf data
  mrf_ind    *edges;
  mrf_data   *weights;
  mrf_data   *dist;
  
  mrf_v_data   unaries_;
  mrf_v_data   h_;
  mrf_v_data   w_;
  mrf_v_data   y_;
  mrf_v_label  x_;
  
  vector<bool> in_grid;
  vector<mrf_ind> h_loads_to_repair_;
  vector<mrf_ind> v_loads_to_repair_;
  
  // Graph
  GraphType4C        *g2;
  
  //
  mrf_nrg             nrg;
  
  // Functions
  void init_dual_variables();
  void expansion(const mrf_label l, const bool grow_sink);
  
  void pre_edit_dual(const mrf_label l);
  void post_edit_primal_dual(const mrf_label l);
  mrf_data triangular_inequality(const mrf_label& l0, const mrf_label& l1,
                                 const mrf_label& l2);
  mrf_nrg compute_primal_nrg();
  
  void repair_horizontal_loads();
  void repair_vertical_loads();
  
  // Neighbors
  int next_higher_mul8(int v){ return ((v-1)/8)*8+8;};
  
  
  mrf_ind node(mrf_ind i, mrf_ind j){ return nodeId(i+1, j+1);}
  mrf_ind nodeId(mrf_ind i, mrf_ind j){
    return (((i>>3)+(j>>3)*RB)<<6) + (i&7)+((j&7)<<3);
  };

  mrf_ind up(mrf_ind v){
    return ((((v) & 0x00000007) == 0 ) ? (v)-57 : (v)-1);};
  mrf_ind down(mrf_ind v){
    return ((((~(v)) & 0x00000007) == 0 ) ? (v)+57 : (v)+1 );};
  
  mrf_ind left(mrf_ind v){
    return ((((v) & 0x00000038) == 0 ) ? (v)-YOFS : (v)-8 );};
  mrf_ind right(mrf_ind v){
    return ((((~(v)) & 0x00000038) == 0 ) ? (v)+YOFS : (v)+8 );};
  
  
  // To simplify notations
  mrf_ind& np(const mrf_ind& pq){return edges[2*pq];};
  mrf_ind& nq(const mrf_ind& pq){return edges[2*pq+1];};
  
  mrf_label& x(const mrf_ind& p){return x_[p];};
  mrf_label& xp(const mrf_ind& pq){return x_[np(pq)];};
  mrf_label& xq(const mrf_ind& pq){return x_[nq(pq)];};
  
  mrf_data& h(const mrf_ind& p, const mrf_ind& l){return h_[p + l*NE];};
  mrf_data& h_xp(const mrf_ind& p){return h_[p + x(p)*NE];};
  
  mrf_data& unaries(const mrf_ind& p, const mrf_ind& l){return unaries_[p + l*NE];};
  
  
  mrf_data& wh(const mrf_ind& p){return w_[p];};
  mrf_data& wv(const mrf_ind& p){return w_[p + NE];};
  
  mrf_data& yh(const mrf_ind& p, const mrf_ind& l){return y_[p + NE*l];};
  mrf_data& yh_xp(const mrf_ind& p){return y_[p + xp(p)*NE];};
  mrf_data& yh_xq(const mrf_ind& p){return y_[p + xq(p)*NE];};
  
  mrf_data& yv(const mrf_ind& p, const mrf_ind& l){return y_[p + NE*l +NE*L];};
  mrf_data& yv_xp(const mrf_ind& p){return y_[p + xp(p)*NE +NE*L];};
  mrf_data& yv_xq(const mrf_ind& p){return y_[p + xq(p)*NE +NE*L];};
  
  mrf_data pairwise_cost(const mrf_ind& pq, const mrf_label& lp,
                         const mrf_label& lq);
  
  mrf_data& dist_cost(const mrf_label& lp, const mrf_label& lq);
  
  //void update_dual(const mrf_label &pq, const mrf_label &lp,
  //                 const mrf_data &delta);
  
  float get_timing(clock_t timer);
  
  
  // Sanity check
  void all_sanity_check();
  void sanity_active_blance();
  void sanity_height_balance();
  void sanity_indices();
  
};

//----------------------------------------------------------------------------//
inline float BC_PD_CF::get_timing(clock_t timer)
{
  return ((float)(clock() - timer))/CLOCKS_PER_SEC;
}

//----------------------------------------------------------------------------//
inline mrf_data BC_PD_CF::pairwise_cost(const mrf_ind& pq,
                                     const mrf_label& lp,
                                     const mrf_label& lq) {
  
  return weights[pq] * dist_cost(lp,lq);
}

//----------------------------------------------------------------------------//
inline mrf_data& BC_PD_CF::dist_cost(const mrf_label& lp, const mrf_label& lq)
{
  return dist[lp+lq*L];
};

//----------------------------------------------------------------------------//
inline mrf_data BC_PD_CF::triangular_inequality(const mrf_label& l0,
                                                const mrf_label& l1,
                                                const mrf_label& l2)
{
  return dist_cost(l0,l1) + dist_cost(l1,l2) - dist_cost(l0,l2);
}


#endif /* bc_pd_cache_friendly_hpp */
