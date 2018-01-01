//
//  bc_pd_cache_friendly.cpp
//  All_Fast_PD
//
//  Created by Bruno CONEJO on 1/7/17.
//  Copyright © 2017 Bruno CONEJO. All rights reserved.
//

#include "bc_pd_cache_friendly.hpp"

//-----------------------------------------------------------------------------//
BC_PD_CF::BC_PD_CF(int R_, int C_, int L_, vector<float> unaries_, vector<float> weights_, vector<float> dist_, vector<float> x_init_)

{
  // Sizes of MRF
  R = R_;
  C = C_;
  L = L_;
  
  N = R*C;

  // Extended sizes for cache friendlyness
  RE =next_higher_mul8(R+2);
  CE =next_higher_mul8(C+2);
  NE = RE * CE;
  RB = RE/8;
  YOFS = (RB-1)*64+8;
  
  
  printf("MRF with %u nodes (%u, %u), %u labels\n",N, R, C, L);
  
  // Pointers to mrf data
  weights = &weights_[0];
  dist.resize(L*L, 0);
  for(mrf_ind a=0; a<L*L; ++a)
    dist[a] = dist_[a];
  
  // Create primal and dual variables
  x_.resize(NE, 0);
  for(mrf_ind j=0; j<C; ++j){
    for(mrf_ind i=0; i<R; ++i){
      mrf_ind p = i + j*R;
      x(node(i, j)) = x_init_[p];
    }
  }
  
  // Heights
  h_.resize(NE*L, 0);
  for(mrf_label l=0; l<L; ++l)
  {
    
    for(mrf_ind j=0; j<C; ++j){
      for(mrf_ind i=0; i<R; ++i){
      
        mrf_ind p = i + j*R;
        h(node(i,j), l) = unaries_[p+l*N];
        //unaries(node(i,j), l) = mrf.unaries[l].val[p]; //For debug only
      }
    }
  }
  
  // Balance
  y_.resize(2*NE*L, 0);

  // Weights
  w_.resize(2*NE, 0);
  // Horizontal weights
  for(mrf_ind j=0; j<C-1; ++j){
    for(mrf_ind i=0; i<R; ++i){
      
      mrf_ind pq = i + j*R;
      wh(node(i, j)) = weights[pq];
      
    }
  }
  
  // Vertical weights
  for(mrf_ind j=0; j<C; ++j){
    for(mrf_ind i=0; i<R; ++i){
      
      mrf_ind pq = R*C + i + j*R;
      wv(node(i, j)) = weights[pq];
    }
  }
  
  //
  in_grid.resize(NE, false);
  for(mrf_ind j=0; j<C; ++j){
    for(mrf_ind i=0; i<R; ++i){
      in_grid[node(i,j)] = true;
    }
  }
  
  h_loads_to_repair_.reserve(N);
  v_loads_to_repair_.reserve(N);
  
  // Create capacities and graph
  g2 = new GraphType4C(R, C);
  //g2 = new GraphType4C_MT(R, C, 1, 2000);
  
  // Init dual variables
  init_dual_variables();
}

//----------------------------------------------------------------------------//
BC_PD_CF::~BC_PD_CF()
{
  delete g2;
}

//----------------------------------------------------------------------------//
void BC_PD_CF::init_dual_variables()
{
  //
  for(mrf_ind p=0; p<NE; p++){
    
    if(in_grid[p]){
      
      // Horizontal variables
      {
        mrf_ind q = right(p);
        
        mrf_label x_p = x(p);
        mrf_label x_q = x(q);
        
        mrf_data delta = wh(p) * dist_cost(x_p, x_q) - (yh(p, x_p) - yh(p, x_q));
        
        if(delta){
          yh(p, x_p)   += delta;
          h(p, x_p)    += delta;
          h(q, x_p)    -= delta;
        }
      }
      
      // Vertical variables
      {
        mrf_ind r = down(p);
        
        mrf_label x_p = x(p);
        mrf_label x_r = x(r);
        
        mrf_data delta = wv(p) * dist_cost(x_p, x_r) - (yv(p, x_p) - yv(p, x_r));
        
        if(delta){
          yv(p, x_p)   += delta;
          h(p, x_p)    += delta;
          h(r, x_p)    -= delta;
        }
      }
      
    }
    
  }
  
  
}

//----------------------------------------------------------------------------//
void BC_PD_CF::optimize(const size_t I_max, bool grow_sink)
{
  
  //all_sanity_check();
  
  nrg = compute_primal_nrg();
  printf("Init: %f\n", nrg);
  
  for(size_t i=0; i<I_max; ++i)
  {
    printf("  Iter %zu: ",i);
    clock_t t_outer = clock();
    
    for(mrf_label l=0; l<L; ++l)
    {
      expansion(l, grow_sink);
      //all_sanity_check();
    }
    
    mrf_nrg nrg_new = compute_primal_nrg();
    printf("%.2f sec, APF = %f \n", get_timing(t_outer),nrg_new);
    
    if(nrg_new < nrg)
      nrg = nrg_new;
    else
      break;;
  }
  
}


//----------------------------------------------------------------------------//
void BC_PD_CF::expansion(const mrf_label l, const bool grow_sink)
{
  
  // Pre edit dual
  pre_edit_dual(l);
  
  // Max-flow
  g2->compute_maxflow(grow_sink);
  
  // Post edit
  post_edit_primal_dual(l);
  
}

//----------------------------------------------------------------------------//
void BC_PD_CF::pre_edit_dual(const mrf_label l)
{
  
  // Reset graph
  g2->reset();
  
  for(mrf_ind p=0; p<NE; p++){
    
    if(in_grid[p]){
      
      mrf_ind q = right(p);
      mrf_ind r = down(p);
      mrf_label x_p = x(p);
      mrf_label x_q = x(q);
      mrf_label x_r = x(r);
      
      // Horizontal edges
      if ( (x_p!=l) & (x_q!=l) )
      {
        mrf_data delta0 = wh(p) * dist_cost(l, x_q) - (yh(p, l) - yh(p, x_q));
        mrf_data delta1 = wh(p) * dist_cost(x_p, l) - (yh(p, x_p) - yh(p, l));
        
        if(delta0<0 | delta1<0)
        {
          yh(p, l)   += delta0;
          h(p, l)    += delta0;
          h(q, l)    -= delta0;
          delta1     += delta0;
          delta0      = 0;
          
          // Non metric distance
          if (triangular_inequality(x_p, l, x_q)<0) {
            delta1 = 0;
            h_loads_to_repair_.push_back(p);
          }
          
          if(delta1<0) // Only for non distance functions?
            delta1 = 0;
          
        }
        
        g2->set_neighbor_cap(p, 0, +1, delta0);
        g2->set_neighbor_cap(q, 0, -1, delta1);
        
        yh(p, l) += delta0;
      }
      
      // Vertical edges
      if ( (x_p!=l) & (x_r!=l) ) {
        
        mrf_data delta0 = wv(p) * dist_cost(l, x_r) - (yv(p, l) - yv(p, x_r));
        mrf_data delta1 = wv(p) * dist_cost(x_p, l) - (yv(p, x_p) - yv(p, l));
        
        if(delta0<0 | delta1<0)
        {
          
          yv(p, l)   += delta0;
          h(p, l)    += delta0;
          h(r, l)    -= delta0;
          delta1     += delta0;
          delta0      = 0;
          
          // Non metric distance
          if (triangular_inequality(x_p, l, x_r)<0) {
            delta1 = 0;
            v_loads_to_repair_.push_back(p);
          }
          
          if(delta1<0) // Only for non distance functions?
            delta1 = 0;
          
        }
        
        g2->set_neighbor_cap(p, +1, 0, delta0);
        g2->set_neighbor_cap(r, -1, 0, delta1);
        
        yv(p, l) += delta0;
        
      }
      
      // Height variables
      mrf_data delta = h_xp(p) - h(p, l);
      g2->set_terminal_cap(p, delta);
      
    }
    
  }
}

//----------------------------------------------------------------------------//
void BC_PD_CF::post_edit_primal_dual(const mrf_label l)
{
  
  for(mrf_ind p=0; p<NE; ++p){
    
    // Height
    mrf_data delta = h_xp(p) - h(p, l);
    //h(p,l) = h_xp(p);
    
    if(delta > 0)
      h(p,l) = h_xp(p) - g2->get_terminal_cap(p);
    else
      h(p,l) = h_xp(p) + g2->get_terminal_cap(p);
    
    // Update label
    if(g2->isSaturated(p))
      x(p) = l;
    
    // Horizontal edges
    yh(p, l) -=g2->get_neighbor_cap(p, 0, +1);
    
    // Vertical edges
    yv(p, l) -= g2->get_neighbor_cap(p, +1, 0);
    
  }
  
  // Repairs load
  repair_horizontal_loads();
  repair_vertical_loads();
  
}


//----------------------------------------------------------------------------//
void BC_PD_CF::repair_horizontal_loads() {
  
  while (!h_loads_to_repair_.empty()) {
    mrf_ind p = h_loads_to_repair_.back();
    h_loads_to_repair_.pop_back();
    
    mrf_ind q = right(p);
    mrf_label x_p = x(p);
    mrf_label x_q = x(q);
    
    if (g2->isSaturated(q)) {
      
      mrf_data dual_f = (yh(p, x_p) - yh(p, x_q)) - wh(p) * dist_cost(x_p, x_q);
      
      if (dual_f>0) {
        yh(p, x_p) -= dual_f;
        h(p, x_p)  -= dual_f;
        h(q, x_p)  += dual_f;
      }
    }
    
    assert( (yh(p, x_p) - yh(p, x_q)) == wh(p) * dist_cost(x_p, x_q) );
  }
}

//----------------------------------------------------------------------------//
void BC_PD_CF::repair_vertical_loads() {
  
  while (!v_loads_to_repair_.empty()) {
    mrf_ind p = v_loads_to_repair_.back();
    v_loads_to_repair_.pop_back();
    
    mrf_ind q = down(p);
    mrf_label x_p = x(p);
    mrf_label x_q = x(q);
    
    if (g2->isSaturated(q)) {
      
      mrf_data dual_f = (yv(p, x_p) - yv(p, x_q)) - wv(p) * dist_cost(x_p, x_q);
      
      if (dual_f>0) {
        yv(p, x_p) -= dual_f;
        h(p, x_p)  -= dual_f;
        h(q, x_p)  += dual_f;
      }
    }
    
    assert( (yv(p, x_p) - yv(p, x_q)) == wv(p) * dist_cost(x_p, x_q) );
  }
}


//----------------------------------------------------------------------------//
mrf_nrg BC_PD_CF::compute_primal_nrg()
{
  mrf_nrg nrg_tmp=0;
  
  for(mrf_ind j=0; j<C; ++j)
    for(mrf_ind i=0; i<R; ++i)
    nrg_tmp += h_xp(node(i,j));
  
  return nrg_tmp;
}

//----------------------------------------------------------------------------//
void BC_PD_CF::restore_unaries() {
  
  for (mrf_label l=0; l<L; ++l) {
    
    // Horizontal edges
    for(mrf_ind j=0; j<C-1; ++j){
      for(mrf_ind i=0; i<R; ++i){
        
        mrf_ind p = node(i, j);
        mrf_ind q = right(p);
        
        h(p, l) -= yh(p, l);
        h(q, l) += yh(q, l);
      }
    }
    
    // Vertical edges
    for(mrf_ind j=0; j<C-1; ++j){
      for(mrf_ind i=0; i<R; ++i){
        
        mrf_ind p = node(i, j);
        mrf_ind q = down(p);
        
        h(p, l) -= yv(p, l);
        h(q, l) += yv(q, l);
      }
    }
    
  }
}

//----------------------------------------------------------------------------//
void BC_PD_CF::all_sanity_check() {
  sanity_indices();
  sanity_active_blance();
  sanity_height_balance();
}

//----------------------------------------------------------------------------//
void BC_PD_CF::sanity_active_blance() {
  
  // Horizontal edges
  for(mrf_ind j=0; j<C-1; ++j){
    for(mrf_ind i=0; i<R; ++i){
      
      mrf_ind p = node(i, j);
      mrf_ind q = right(p);
      
      mrf_label x_p = x(p);
      mrf_label x_q = x(q);
      
      mrf_data delta = wh(p) * dist_cost(x_p, x_q) - (yh(p, x_p) - yh(p, x_q));
      assert( delta == 0 );
    }
  }
  
  // Vertical edges
  for(mrf_ind j=0; j<C; ++j){
    for(mrf_ind i=0; i<R-1; ++i){
      
      mrf_ind p = node(i, j);
      mrf_ind q = down(p);
      
      mrf_label x_p = x(p);
      mrf_label x_q = x(q);
      
      mrf_data delta = wv(p) * dist_cost(x_p, x_q) - (yv(p, x_p) - yv(p, x_q));
      assert( delta == 0 );
    }
  }
  
}

//----------------------------------------------------------------------------//
// Check hp(l) = up(l) + sum_q ypq(l)
void BC_PD_CF::sanity_height_balance() {
  
  for (mrf_ind l=0; l<L; ++l) {
    vector<mrf_data> h_tmp(NE,0);
    
    
    // Horizontal edges
    for(mrf_ind j=0; j<C-1; ++j){
      for(mrf_ind i=0; i<R; ++i){
        
        mrf_ind p = node(i, j);
        mrf_ind q = right(p);
        
        h_tmp[p] += yh(p, l);
        h_tmp[q] -= yh(q, l);
      }
    }
    
    // Vertical edges
    for(mrf_ind j=0; j<C; ++j){
      for(mrf_ind i=0; i<R-1; ++i){
        
        mrf_ind p = node(i, j);
        mrf_ind q = down(p);
        
        h_tmp[p] += yv(p, l);
        h_tmp[q] -= yv(q, l);
      }
    }
    
    for(mrf_ind j=0; j<C; ++j){
      for(mrf_ind i=0; i<R; ++i) {
        mrf_ind p = node(i, j);
        h_tmp[p] += unaries(p, l); // Add up(l)
        
        mrf_data delta = h_tmp[p] - h(p, l);
        assert(delta == 0);
      }
    }
  }
}

//----------------------------------------------------------------------------//
// Check left right up and down operator
void BC_PD_CF::sanity_indices() {

  for(mrf_ind j=0; j<C; ++j){
    for(mrf_ind i=0; i<R; ++i){
      
      mrf_ind p = node(i, j);
      
      assert(node(i, j+1) == right(p));
      assert(node(i, j-1) == left(p));
      
      assert(node(i-1, j) == up(p));
      assert(node(i+1, j) == down(p));
      
    }
  }
}

//----------------------------------------------------------------------------//
mrf_v_label BC_PD_CF::get_solution(){
  mrf_v_label x_out(N);
  for(mrf_ind j=0; j<C; ++j) {
    for(mrf_ind i=0; i<R; ++i) {
      mrf_ind p = i + j*R;
      x_out[p] = x(node(i, j));
    }
  }
  return x_out;
};

//----------------------------------------------------------------------------//
void BC_PD_CF::get_dual_variables(double *h_out, double *yh_out, double *yv_out)
{

  for (mrf_ind l=0; l<L; ++l) {
    
    for(mrf_ind j=0; j<C; ++j){
      for(mrf_ind i=0; i<R; ++i){
        
        mrf_ind p = i + j*R;
        
        h_out[p+l*N] = h(node(i, j), l);
        yv_out[p+l*N] = yh(node(i, j), l);
        yh_out[p+l*N] = yv(node(i, j), l);
        
      }
    }
  }

}
