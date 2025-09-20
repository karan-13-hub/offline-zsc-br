import torch
import torch.nn as nn
from typing import Tuple, Dict
import pdb
from functools import partial

def td_error(
    agent_1,
    agent_2,
    obs: Dict[str, torch.Tensor],
    hid: Dict[str, torch.Tensor],
    action: Dict[str, torch.Tensor],
    reward: torch.Tensor,
    terminal: torch.Tensor,
    bootstrap: torch.Tensor,
    seq_len: torch.Tensor,
    args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    max_seq_len = obs["priv_s"].size(0)
    priv_s = obs["priv_s"]
    publ_s = obs["publ_s"]
    legal_move = obs["legal_move"]
    action = action["a"]
    
    new_hid = {}
    for k, v in hid.items():
        new_hid[k] = v.clone().flatten(1, 2).contiguous()

    bsize = priv_s.size(1)
    num_player = priv_s.size(2)
    priv_s = priv_s.flatten(1, 2)
    publ_s = publ_s.flatten(1, 2)
    legal_move = legal_move.flatten(1, 2)
    action = action.flatten(1, 2)
    
    # this only works because the trajectories are padded,
    # i.e. no terminal in the middle
    ag1_online_qa, ag1_greedy_a, ag1_online_q, _ = agent_1.online_net(
        priv_s, publ_s, legal_move, action, new_hid
    )

    ag1_target_qa, _, ag1_target_q, _ = agent_1.target_net(
        priv_s, publ_s, legal_move, ag1_greedy_a, new_hid
    )

    ag2_online_qa, ag2_greedy_a, ag2_online_q, _ = agent_2.online_net(
        priv_s, publ_s, legal_move, action, new_hid
    )

    ag2_target_qa, _, ag2_target_q, _ = agent_2.target_net(
        priv_s, publ_s, legal_move, ag2_greedy_a, new_hid
    )

    #detach agent 2's q values
    ag2_online_qa = ag2_online_qa.detach()
    ag2_online_q = ag2_online_q.detach()
    ag2_greedy_a = ag2_greedy_a.detach()

    # Reshape the Q-values
    ag1_online_qa = ag1_online_qa.view(max_seq_len, bsize, num_player)
    ag2_online_qa = ag2_online_qa.view(max_seq_len, bsize, num_player)
    ag1_online_q = ag1_online_q.view(max_seq_len, bsize, num_player, -1)
    ag2_online_q = ag2_online_q.view(max_seq_len, bsize, num_player, -1)
    
    # Reshape the target Q-values
    ag1_target_qa = ag1_target_qa.view(max_seq_len, bsize, num_player)
    ag2_target_qa = ag2_target_qa.view(max_seq_len, bsize, num_player)
    ag1_target_q = ag1_target_q.view(max_seq_len, bsize, num_player, -1)
    ag2_target_q = ag2_target_q.view(max_seq_len, bsize, num_player, -1)
    
    # Get the action dimension from the Q-values
    action_dim = ag1_online_q.size(-1)
    
    # Create masks for alternating between agents
    # For even indices (0, 2, 4, ...) use agent 1, for odd indices (1, 3, 5, ...) use agent 2
    even_mask = torch.zeros(max_seq_len, bsize, num_player, action_dim, device=ag1_online_q.device)
    odd_mask = torch.zeros(max_seq_len, bsize, num_player, action_dim, device=ag1_online_q.device)
    
    # Create masks for each player
    for i in range(num_player):
        if i % 2 == 0:
            even_mask[:, :, i, :] = 1
        else:
            odd_mask[:, :, i, :] = 1
    
    # Create cross-play Q-values for agent 1 as player 1, agent 2 as player 2
    online_qa_1_2 = (ag1_online_qa * even_mask[:, :, :, 0] + ag2_online_qa * odd_mask[:, :, :, 0]).sum(-1)
    # online_q_1_2 = (ag1_online_q * even_mask + ag2_online_q * odd_mask)
    target_qa_1_2 = (ag1_target_qa * even_mask[:, :, :, 0] + ag2_target_qa * odd_mask[:, :, :, 0]).sum(-1)
    # target_q_1_2 = (ag1_target_q * even_mask + ag2_target_q * odd_mask)
    
    # Create cross-play Q-values for agent 2 as player 1, agent 1 as player 2
    online_qa_2_1 = (ag2_online_qa * even_mask[:, :, :, 0] + ag1_online_qa * odd_mask[:, :, :, 0]).sum(-1)
    # online_q_2_1 = (ag2_online_q * even_mask + ag1_online_q * odd_mask)
    target_qa_2_1 = (ag2_target_qa * even_mask[:, :, :, 0] + ag1_target_qa * odd_mask[:, :, :, 0]).sum(-1)
    # target_q_2_1 = (ag2_target_q * even_mask + ag1_target_q * odd_mask)
    
    # Process target Q-values for both agent combinations
    target_qa_1_2 = torch.cat(
        [target_qa_1_2[args.multi_step :], target_qa_1_2[: args.multi_step]], 0
    )
    target_qa_1_2[-args.multi_step :] = 0
    
    target_qa_2_1 = torch.cat(
        [target_qa_2_1[args.multi_step :], target_qa_2_1[: args.multi_step]], 0
    )
    target_qa_2_1[-args.multi_step :] = 0
    
    # Create target values for both agent combinations
    target_1_2 = reward + bootstrap * (args.gamma ** args.multi_step) * target_qa_1_2
    target_2_1 = reward + bootstrap * (args.gamma ** args.multi_step) * target_qa_2_1
    
    # Verify target sizes
    assert target_1_2.size() == reward.size()
    assert target_2_1.size() == reward.size()
    
    # Create mask for valid sequences
    mask = torch.arange(0, max_seq_len, device=seq_len.device)
    mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
    
    # Calculate error for both agent combinations
    err_1_2 = (target_1_2.detach() - online_qa_1_2) * mask
    err_2_1 = (target_2_1.detach() - online_qa_2_1) * mask
    
    # Compute policy diversity between online_q_1_2 and online_q_2_1 at specific indices
    # Create indices for (0,0), (1,1), (2,0), (3,1), (4,0), (5,1), ...
    # More efficient: use torch.arange instead of loops
    seq_indices = torch.arange(max_seq_len, device=ag1_online_q.device)
    player_indices = seq_indices % 2  # 0 for even indices, 1 for odd indices
    
    # Extract Q-values at the specified indices - more efficient indexing
    # Do not apply the mask to the Q-values
    ag1_online_q_diversity = ag1_online_q[seq_indices, :, player_indices, :]
    ag2_online_q_diversity = ag2_online_q[seq_indices, :, player_indices, :]
    
    # Return both sets of values and the errors
    return err_1_2, err_2_1, ag1_online_q_diversity, ag2_online_q_diversity, mask


def td_error_br(
    agent,
    agent_br,
    obs: Dict[str, torch.Tensor],
    hid: Dict[str, torch.Tensor],
    action: Dict[str, torch.Tensor],
    reward: torch.Tensor,
    terminal: torch.Tensor,
    bootstrap: torch.Tensor,
    seq_len: torch.Tensor,
    args
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    max_seq_len = obs["priv_s"].size(0)
    priv_s = obs["priv_s"]
    publ_s = obs["publ_s"]
    legal_move = obs["legal_move"]
    action = action["a"]
    
    bsize = priv_s.size(1)
    num_player = priv_s.size(2)
    priv_s = priv_s.flatten(1, 2)
    publ_s = publ_s.flatten(1, 2)
    legal_move = legal_move.flatten(1, 2)
    action = action.flatten(1, 2)
    
    # this only works because the trajectories are padded,
    # i.e. no terminal in the middle
    new_hid = get_new_hid(hid)
    ag1_online_qa, ag1_greedy_a, ag1_online_q, _ = agent.online_net(
        priv_s, publ_s, legal_move, action, new_hid
    )

    new_hid = get_new_hid(hid)
    ag1_target_qa, _, ag1_target_q, _ = agent.target_net(
        priv_s, publ_s, legal_move, ag1_greedy_a, new_hid
    )

    new_hid = get_new_hid(hid)
    ag2_online_qa, ag2_greedy_a, ag2_online_q, _ = agent_br.online_net(
        priv_s, publ_s, legal_move, action, new_hid
    )
    
    new_hid = get_new_hid(hid)
    ag2_target_qa, _, _, _ = agent_br.target_net(
        priv_s, publ_s, legal_move, ag2_greedy_a, new_hid
    )

    #detach agent's q values
    ag1_online_qa = ag1_online_qa.detach()
    ag1_online_q_div = ag1_online_q.clone()
    ag1_online_q = ag1_online_q.detach()
    ag1_target_qa = ag1_target_qa.detach()

    # Reshape the Q-values
    ag1_online_qa = ag1_online_qa.view(max_seq_len, bsize, num_player)
    ag2_online_qa = ag2_online_qa.view(max_seq_len, bsize, num_player)
    ag1_online_q = ag1_online_q.view(max_seq_len, bsize, num_player, -1)
    ag2_online_q = ag2_online_q.view(max_seq_len, bsize, num_player, -1)
    ag1_online_q_div = ag1_online_q_div.view(max_seq_len, bsize, num_player, -1)

    # Reshape the target Q-values
    ag1_target_qa = ag1_target_qa.view(max_seq_len, bsize, num_player)
    ag2_target_qa = ag2_target_qa.view(max_seq_len, bsize, num_player)
    # ag1_target_q = ag1_target_q.view(max_seq_len, bsize, num_player, -1)
    # ag2_target_q = ag2_target_q.view(max_seq_len, bsize, num_player, -1)
    
    # Get the action dimension from the Q-values
    action_dim = ag1_online_q.size(-1)
    
    # Create masks for alternating between agents
    # For even indices (0, 2, 4, ...) use agent 1, for odd indices (1, 3, 5, ...) use agent 2
    even_mask = torch.zeros(max_seq_len, bsize, num_player, action_dim, device=ag1_online_q.device)
    odd_mask = torch.zeros(max_seq_len, bsize, num_player, action_dim, device=ag1_online_q.device)
    
    # Create masks for each player
    for i in range(num_player):
        if i % 2 == 0:
            even_mask[:, :, i, :] = 1
        else:
            odd_mask[:, :, i, :] = 1
    
    # Create cross-play Q-values for agent 1 as player 1, agent 2 as player 2
    online_qa_1_2 = (ag1_online_qa * even_mask[:, :, :, 0] + ag2_online_qa * odd_mask[:, :, :, 0]).sum(-1)
    target_qa_1_2 = (ag1_target_qa * even_mask[:, :, :, 0] + ag2_target_qa * odd_mask[:, :, :, 0]).sum(-1)

    online_q_1_2 = (ag1_online_q * even_mask[:, :, :, :] + ag2_online_q * odd_mask[:, :, :, :])
    online_q_1_2 = online_q_1_2.view(max_seq_len, bsize*num_player, -1)
    
    # Process target Q-values for both agent combinations
    target_qa_1_2 = torch.cat(
        [target_qa_1_2[args.multi_step :], target_qa_1_2[: args.multi_step]], 0
    )
    target_qa_1_2[-args.multi_step :] = 0
    
    # Create target values for both agent combinations
    target_1_2 = reward + bootstrap * (args.gamma ** args.multi_step) * target_qa_1_2
    
    # Verify target sizes
    assert target_1_2.size() == reward.size()
    
    # Create mask for valid sequences
    mask = torch.arange(0, max_seq_len, device=seq_len.device)
    mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
    
    # Calculate error for both agent combinations
    err_1_2 = (target_1_2.detach() - online_qa_1_2) * mask
    
    # Compute policy diversity between online_q_1_2 and online_q_2_1 at specific indices
    # Create indices for (0,0), (1,1), (2,0), (3,1), (4,0), (5,1), ...
    # More efficient: use torch.arange instead of loops
    seq_indices = torch.arange(max_seq_len, device=ag1_online_q.device)
    player_indices = seq_indices % 2  # 0 for even indices, 1 for odd indices
    
    # Extract Q-values at the specified indices - more efficient indexing
    # Do not apply the mask to the Q-values
    ag1_online_q_diversity = ag1_online_q_div[seq_indices, :, player_indices, :]
    if args.bc:
        legal_q = (1 + online_q_1_2 - online_q_1_2.min()) * legal_move
        legal_logit = nn.functional.softmax(legal_q, -1)
        gt_logit = torch.zeros_like(legal_logit, dtype=torch.float32)
        gt_logit.scatter_(2, action.unsqueeze(-1), 1)
        gt_logit = gt_logit.view(-1, online_q_1_2.shape[-1])
        legal_logit = legal_logit.view(-1, online_q_1_2.shape[-1])
        bc_loss = torch.nn.functional.cross_entropy(legal_logit, gt_logit, reduction="none")
        if args.method == "vdn":
            bc_loss = bc_loss.view(max_seq_len, bsize, num_player).sum(-1)
    else :
        bc_loss = torch.zeros(1, device=err_1_2.device)
    
    if args.cql:
        online_q_1_2 = online_q_1_2.view(max_seq_len, bsize, num_player, -1)
        logsumexp = torch.logsumexp(online_q_1_2, dim=2)
        cql_loss = (logsumexp - online_qa_1_2)
    else:
        cql_loss = torch.zeros(1, device=err_1_2.device)
    # Return both sets of values and the errors
    return err_1_2, bc_loss, cql_loss

def cp_loss(agents_1, agents_2, batch, stat, args):
    err_1_2, err_2_1, ag1_online_q_diversity, ag2_online_q_diversity, valid_mask = td_error(
        agents_1,
        agents_2,
        batch.obs,
        batch.h0,
        batch.action,
        batch.reward,
        batch.terminal,
        batch.bootstrap,
        batch.seq_len,
        args,
    )

    err = err_1_2 + err_2_1
    rl_loss = nn.functional.smooth_l1_loss(
        err, torch.zeros_like(err), reduction="none"
    )
    rl_loss = rl_loss.sum(0)
    # stat["rl_loss"].feed((rl_loss / batch.seq_len).mean().item())

    loss = rl_loss

    return loss, ag1_online_q_diversity, ag2_online_q_diversity, valid_mask

def get_new_hid(hid):
    new_hid = {}
    for k, v in hid.items():
        new_hid[k] = v.clone().flatten(1, 2).contiguous()
    return new_hid

def q_ensemble_loss(agent_br, agents, agent_weights, batch, args):
    obs = batch.obs
    h0 = batch.h0
    action = batch.action
    reward = batch.reward
    terminal = batch.terminal
    bootstrap = batch.bootstrap
    seq_len = batch.seq_len

    max_seq_len = obs["priv_s"].size(0)
    priv_s = obs["priv_s"]
    publ_s = obs["publ_s"]
    legal_move = obs["legal_move"]
    action = action["a"]
    
    bsize = priv_s.size(1)
    num_player = priv_s.size(2)
    priv_s = priv_s.flatten(1, 2)
    publ_s = publ_s.flatten(1, 2)
    legal_move = legal_move.flatten(1, 2)
    action = action.flatten(1, 2)
    
    # this only works because the trajectories are padded,
    # i.e. no terminal in the middle
    new_hid = get_new_hid(h0)
    agbr_online_qa, agbr_greedy_a, _, _ = agent_br.online_net(
        priv_s, publ_s, legal_move, action, new_hid
    )

    new_hid = get_new_hid(h0)
    agbr_target_qa, _, _, _ = agent_br.target_net(
        priv_s, publ_s, legal_move, agbr_greedy_a, new_hid
    )

    agbr_online_qa = agbr_online_qa.view(max_seq_len, bsize, num_player).sum(-1)
    agbr_target_qa = agbr_target_qa.view(max_seq_len, bsize, num_player).sum(-1)

    af_target_qa = torch.zeros(max_seq_len, bsize, device=args.train_device)
    for i, agent in enumerate(agents):
        agent.eval()
        with torch.no_grad():
            new_hid = get_new_hid(h0)
            _, agent_greedy_a, _, _ = agent.online_net(
                priv_s, publ_s, legal_move, action, new_hid
            )
            new_hid = get_new_hid(h0)
            agent_target_qa, _, agent_target_q, _ = agent.target_net(
                priv_s, publ_s, legal_move, agent_greedy_a, new_hid)
            
            agent_target_qa = agent_target_qa.view(max_seq_len, bsize, num_player).sum(-1)
            
            #normalize the target q value
            agent_target_qa = agent_target_qa / agent_target_qa.norm(dim=-1, keepdim=True)

            af_target_qa = af_target_qa + agent_weights[i] * agent_target_qa
    
    af_target_qa = torch.cat(
        [af_target_qa[args.multi_step :], af_target_qa[: args.multi_step]], 0
    )
    af_target_qa[-args.multi_step :] = 0
    
    assert af_target_qa.size() == reward.size()
    target = reward + bootstrap * (args.gamma ** args.multi_step) * af_target_qa
    
    mask = torch.arange(0, max_seq_len, device=seq_len.device)
    mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
    err = (target.detach() - agbr_online_qa) * mask

    rl_loss = nn.functional.smooth_l1_loss(
        err, torch.zeros_like(err), reduction="none"
    )
    rl_loss = rl_loss.sum(0)    
    return rl_loss

def train_br_agent(agent_br, agents, agent_weights, batch, args):   
    cp_loss = torch.tensor(0.0, device=args.train_device)
    for i, agent in enumerate(agents):
        err, bc_loss, cql_loss = td_error_br(
            agent,
            agent_br,
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            args,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        bc_loss = bc_loss.sum(0)
        cql_loss = cql_loss.sum(0)   
        cp_loss = cp_loss + agent_weights[i] * rl_loss
    cp_loss = cp_loss + args.cp_bc_weight * bc_loss + args.cp_cql_weight * cql_loss
    return cp_loss

def train_br_agent_parallel_manual(agent_br, agents, agent_weights, batch, args):
    """
    Alternative parallelized version using manual parallel processing.
    This version processes agents in parallel using multiprocessing or concurrent.futures.
    """
    import concurrent.futures
    import threading
    
    def process_single_agent(agent):
        err, bc_loss, cql_loss = td_error_br(
            agent,
            agent_br,
            batch.obs,
            batch.h0,
            batch.action,
            batch.reward,
            batch.terminal,
            batch.bootstrap,
            batch.seq_len,
            args,
        )
        rl_loss = nn.functional.smooth_l1_loss(
            err, torch.zeros_like(err), reduction="none"
        )
        rl_loss = rl_loss.sum(0)
        bc_loss = bc_loss.sum(0)
        cql_loss = cql_loss.sum(0)
        return rl_loss, bc_loss, cql_loss
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agents), 4)) as executor:
        # Submit all tasks
        future_to_agent = {executor.submit(process_single_agent, agent): i 
                          for i, agent in enumerate(agents)}
        
        # Collect results
        rl_losses = []
        bc_losses = []
        cql_losses = []
        
        for future in concurrent.futures.as_completed(future_to_agent):
            rl_loss, bc_loss, cql_loss = future.result()
            rl_losses.append(rl_loss)
            bc_losses.append(bc_loss)
            cql_losses.append(cql_loss)
    
    # Calculate weighted sum
    cp_loss = torch.tensor(0.0, device=args.train_device)
    for i, (rl_loss, bc_loss, cql_loss) in enumerate(zip(rl_losses, bc_losses, cql_losses)):
        cp_loss = cp_loss + agent_weights[i] * rl_loss
    
    # Add BC and CQL losses (using the last agent's losses)
    cp_loss = cp_loss + args.cp_bc_weight * bc_losses[-1] + args.cp_cql_weight * cql_losses[-1]
    
    return cp_loss


def diversity_loss_single_agent(agent_q_values, other_q_values, mask, args):
    """
    Calculate diversity loss for a single agent against all other agents.
    
    Args:
        agent_q_values: Q-value tensor for the current agent
        other_q_values: List of Q-value tensors for other agents
        mask: valid action mask for all agents
        args: Arguments containing diversity loss configuration
        
    Returns:
        diversity_loss: Scalar tensor containing the calculated diversity loss for this agent
    """
    # Check if we have at least one other agent to compare with
    if len(other_q_values) < 1 or agent_q_values is None:
        return torch.tensor(0.0, device=agent_q_values.device)
    
    # For JSD (Jensen-Shannon Divergence), calculate divergence for this agent
    if args.div_type == 'jsd':
        # Only select Q-values that are not None
        valid_other_q_values = [q for q in other_q_values if q is not None]
        if len(valid_other_q_values) < 1:
            return torch.tensor(0.0, device=agent_q_values.device)
            
        # Convert all Q-values to probability distributions
        all_q_values = [agent_q_values] + valid_other_q_values
        q_probs = torch.stack([nn.functional.softmax(q, dim=-1) for q in all_q_values], dim=0)
        q_log_probs = torch.stack([nn.functional.log_softmax(q, dim=-1) for q in all_q_values], dim=0)
        
        # Calculate the average probability distribution (detached to avoid gradient issues)
        avg_probs = q_probs.detach().mean(dim=0)  # Average of all agents
        avg_log = torch.log(avg_probs + 1e-10)  # Add small epsilon to avoid log(0)

        # Calculate KL divergence from this agent's distribution to the average of others
        jsd_loss = ((q_log_probs[0] - avg_log) * q_probs[0]).sum(dim=-1)
        jsd_loss = (jsd_loss * mask).sum(dim=0)  # Apply mask and sum over sequence
        
        return jsd_loss

def diversity_loss(q_values, mask, args):
    """
    Calculate diversity loss between Q-values from different agents.
    
    Args:
        q_values: List of Q-value tensors from different agents
        mask: valid action mask for all agents
        args: Arguments containing diversity loss configuration
        
    Returns:
        diversity_loss: Tensor containing the calculated diversity loss for each agent
    """
    # Check if we have at least two Q-value tensors to compare
    if len(q_values) < 2 or q_values[0] is None:
        return torch.tensor(0.0, device=q_values[0].device if q_values[0] is not None else "cpu")
    
    # Get the action dimension from the first Q-value tensor
    action_dim = q_values[0].size(-1)
    
    # For JSD (Jensen-Shannon Divergence), calculate divergence for each agent
    if args.div_type == 'jsd':
        # Only select Q-values that are not None
        valid_q_values = [q for q in q_values if q is not None]
        if len(valid_q_values) < 2:
            return torch.tensor(0.0, device=q_values[0].device)
            
        # Convert all Q-values to probability distributions
        q_probs = torch.stack([nn.functional.softmax(q, dim=-1) for q in valid_q_values], dim=0)
        q_log_probs = torch.stack([nn.functional.log_softmax(q, dim=-1) for q in valid_q_values], dim=0)
        
        # Calculate the average probability distribution (detached to avoid gradient issues)
        avg_probs = q_probs.clone().detach().mean(dim=0)
        avg_log = torch.log(avg_probs + 1e-10)  # Add small epsilon to avoid log(0)

        # Calculate JSD loss for each agent independently
        jsd_losses = []
        for i in range(len(valid_q_values)):
            # Calculate KL divergence from this agent's distribution to the average
            jsd_loss = ((q_log_probs[i] - avg_log) * q_probs[i]).sum(dim=-1)
            jsd_loss = (jsd_loss * mask).sum(dim=0)  # Apply mask and sum over sequence
            jsd_losses.append(jsd_loss)
        
        return torch.stack(jsd_losses)

    # For other diversity types, use pairwise approach
    # Initialize the total diversity loss for each agent
    num_agents = len(q_values)
    total_diversity_loss = torch.zeros(num_agents, device=q_values[0].device)
    
    # Pre-compute probability distributions and log probabilities
    q_probs = []
    q_log_probs = []
    for q in q_values:
        if q is not None:
            q_probs.append(nn.functional.softmax(q, dim=-1))
            q_log_probs.append(nn.functional.log_softmax(q, dim=-1))
        else:
            q_probs.append(None)
            q_log_probs.append(None)
    
    # Calculate diversity between all pairs of agents
    for i in range(num_agents):
        if q_values[i] is None:
            continue
            
        agent_q = q_values[i]
        agent_probs = q_probs[i]
        agent_log_probs = q_log_probs[i]
        
        for j in range(i + 1, num_agents):
            if q_values[j] is None:
                continue
                
            other_q = q_values[j]
            other_probs = q_probs[j]
            other_log_probs = q_log_probs[j]
            
            # Select the type of diversity loss
            if args.div_type == 'ce':
                # Cross-entropy loss
                ce_loss = -(agent_log_probs * other_probs).sum(dim=-1)
                ce_loss = ce_loss + -(other_log_probs * agent_probs).sum(dim=-1)
                pair_diversity_loss = ce_loss
                
            elif args.div_type == 'kl':
                # KL Divergence
                kl_div_1_2 = nn.functional.kl_div(
                    agent_log_probs, other_probs, reduction='none'
                ).sum(dim=-1)
                kl_div_2_1 = nn.functional.kl_div(
                    other_log_probs, agent_probs, reduction='none'
                ).sum(dim=-1)
                pair_diversity_loss = kl_div_1_2 + kl_div_2_1
                
            elif args.div_type == 'cosine':
                # Cosine Similarity
                agent_flat = agent_q.reshape(-1, action_dim)
                other_flat = other_q.reshape(-1, action_dim)
                
                # Normalize the vectors
                agent_norm = agent_flat / (agent_flat.norm(dim=1, keepdim=True) + 1e-10)
                other_norm = other_flat / (other_flat.norm(dim=1, keepdim=True) + 1e-10)
                
                # Compute cosine similarity
                cosine_sim = (agent_norm * other_norm).sum(dim=1)
                pair_diversity_loss = 1.0 - cosine_sim
                
            elif args.div_type == 'l2':
                # L2 Distance
                pair_diversity_loss = torch.norm(agent_q - other_q, p=2, dim=-1)
                
            elif args.div_type == 'combined':
                # Combined approach using multiple metrics with weights
                
                # 1. Cross-entropy loss
                ce_loss = -(agent_log_probs * other_probs).sum(dim=-1)
                ce_loss = ce_loss + -(other_log_probs * agent_probs).sum(dim=-1)
                
                # 2. KL Divergence
                kl_div_1_2 = nn.functional.kl_div(
                    agent_log_probs, other_probs, reduction='none'
                ).sum(dim=-1)
                kl_div_2_1 = nn.functional.kl_div(
                    other_log_probs, agent_probs, reduction='none'
                ).sum(dim=-1)
                kl_div_loss = kl_div_1_2 + kl_div_2_1
                
                # 3. Jensen-Shannon Divergence
                m = 0.5 * (agent_probs + other_probs)
                m_log = torch.log(m + 1e-10)
                
                jsd_loss = 0.5 * (
                    (agent_log_probs - m_log) * agent_probs
                ).sum(dim=-1) + 0.5 * (
                    (other_log_probs - m_log) * other_probs
                ).sum(dim=-1)
                
                # 4. Cosine Similarity
                agent_flat = agent_q.reshape(-1, action_dim)
                other_flat = other_q.reshape(-1, action_dim)
                
                agent_norm = agent_flat / (agent_flat.norm(dim=1, keepdim=True) + 1e-10)
                other_norm = other_flat / (other_flat.norm(dim=1, keepdim=True) + 1e-10)
                
                cosine_sim = (agent_norm * other_norm).sum(dim=1)
                cosine_loss = 1.0 - cosine_sim
                
                # 5. L2 Distance
                l2_dist = torch.norm(agent_q - other_q, p=2, dim=-1)
                
                # Combine all diversity metrics with weights
                pair_diversity_loss = (
                    getattr(args, 'ce_weight', 1.0) * ce_loss +
                    getattr(args, 'kl_weight', 1.0) * kl_div_loss +
                    getattr(args, 'jsd_weight', 1.0) * jsd_loss +
                    getattr(args, 'cosine_weight', 1.0) * cosine_loss +
                    getattr(args, 'l2_weight', 1.0) * l2_dist
                )
            else:
                # Default to cross-entropy if div_type is not recognized
                ce_loss = -(agent_log_probs * other_probs).sum(dim=-1)
                ce_loss = ce_loss + -(other_log_probs * agent_probs).sum(dim=-1)
                pair_diversity_loss = ce_loss
            
            # Apply valid mask to the pair's diversity loss
            pair_diversity_loss = (pair_diversity_loss * mask).mean()
            
            # Add this pair's diversity loss to both agents
            total_diversity_loss[i] += pair_diversity_loss
            total_diversity_loss[j] += pair_diversity_loss
    
    return total_diversity_loss