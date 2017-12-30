//
//  main.cpp
//  Fast-pd_grid_cut
//
//  Created by Bruno CONEJO on 12/27/17.
//  Copyright © 2017 Bruno Conejo. All rights reserved.
//

#include <iostream>
#include "bc_pd_cache_friendly.hpp"

mrf_v_data create_unaries(const int N, const int L){

  mrf_v_data unaries(N*L, 10);

  const int l_opt = L/2;
  for(int n=0; n<N; n++)
    unaries[n + l_opt*N] = 0;

  return unaries;

}

mrf_v_data create_l1_dist(const int L){

  mrf_v_data dist(L*L);
  for(int l0=0; l0<L; l0++){
    for(int l1=0; l1<L; l1++){
      dist[l0 + l1*L] = std::abs(l0-l1);
    }
  }
  
  return dist;
  
}


int main(int argc, const char * argv[]) {
  
  const int R=10, C=20, L=5;
  const int N=R*C;
  const int I_max = 10;
  
  mrf_v_data weights(R*C*2, 1);
  mrf_v_data x_init(N, 0);
  
  mrf_v_data unaries = create_unaries(N, L);
  mrf_v_data dist = create_l1_dist(L);
  
  BC_PD_CF solver = BC_PD_CF(R, C, L, unaries.data(), weights.data(), dist.data(), x_init.data());
  solver.optimize(I_max, false);
  solver.restore_unaries();
  
  std::cout << "We are done!\n";
  return 0;
}
