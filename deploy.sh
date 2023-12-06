python run_mab.py -domain=KdV -trunk=default -heuristic=ucb -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ucb -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ucb -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ucb -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ucb -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=default -heuristic=ts -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ts -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ts -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ts -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=ts -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=default -heuristic=explore -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=explore -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=explore -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=explore -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=explore -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=default -heuristic=rnd -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=rnd -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=rnd -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=rnd -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=default -heuristic=rnd -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 


python run_mab.py -domain=KdV -trunk=stage -heuristic=ucb -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ucb -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ucb -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ucb -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ucb -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=stage -heuristic=ts -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ts -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ts -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ts -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=ts -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=stage -heuristic=explore -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=explore -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=explore -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=explore -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=explore -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=KdV -trunk=stage -heuristic=rnd -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=rnd -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=rnd -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=rnd -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=KdV -trunk=stage -heuristic=rnd -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 



python run_pinn.py -domain=KdV -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 




#=============================================



python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 


python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=50000 -int_adam_test=1000 -int_lbfgs_test=50000 -device=cuda:0 


python run_pinn.py -domain=Burgers -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=2000 -int_lbfgs_test=50000 -device=cuda:0 


#=============================================


python run_mab.py -domain=Advec -trunk=default -heuristic=ucb -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ucb -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ucb -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ucb -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ucb -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=default -heuristic=ts -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ts -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ts -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ts -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=ts -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=default -heuristic=explore -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=explore -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=explore -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=explore -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=explore -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=default -heuristic=rnd -mode=query -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=rnd -mode=batch -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=rnd -mode=step -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=rnd -mode=step2 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=default -heuristic=rnd -mode=step3 -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=stage -heuristic=ucb -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ucb -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ucb -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ucb -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ucb -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=stage -heuristic=ts -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ts -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ts -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ts -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=ts -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=stage -heuristic=explore -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=explore -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=explore -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=explore -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=explore -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 

python run_mab.py -domain=Advec -trunk=stage -heuristic=rnd -mode=query -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=rnd -mode=batch -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=rnd -mode=step -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=rnd -mode=step2 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 
python run_mab.py -domain=Advec -trunk=stage -heuristic=rnd -mode=step3 -num_adam=2 -int_adam=1000 -num_lbfgs=2 -int_lbfgs=10000 -int_adam_test=1000 -int_lbfgs_test=10000 -device=cuda:0 


python run_pinn.py -domain=Advec -num_adam=1 -int_adam=2000 -num_lbfgs=1 -int_lbfgs=20000 -int_adam_test=2000 -int_lbfgs_test=20000 -device=cuda:0 



#================================================


python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=query -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=batch -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step2 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ucb -mode=step3 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=query -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=batch -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step2 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=ts -mode=step3 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=query -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=batch -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step2 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=explore -mode=step3 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=query -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=batch -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step2 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=default -heuristic=rnd -mode=step3 -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=query -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=batch -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step2 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ucb -mode=step3 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=query -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=batch -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step2 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=ts -mode=step3 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=query -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=batch -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step2 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=explore -mode=step3 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 

python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=query -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=batch -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step2 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 
python run_mab.py -domain=Burgers -trunk=stage -heuristic=rnd -mode=step3 -num_adam=2 -int_adam=5000 -num_lbfgs=2 -int_lbfgs=25000 -int_adam_test=5000 -int_lbfgs_test=25000 -device=cuda:0 

python run_pinn.py -domain=Burgers -num_adam=1 -int_adam=10000 -num_lbfgs=1 -int_lbfgs=50000 -int_adam_test=10000 -int_lbfgs_test=50000 -device=cuda:0 




