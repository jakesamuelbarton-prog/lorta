
#add quote marks!!!!!



import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import warnings


class LoRTAPreconditioner:
    """
    Exact implementation of LoRTA preconditioners following URSS  6.1-6.5

    Implements preconditioners for the 5 tensor parameters:
    - A ∈ R^(d×r) - shared embedding factors
    - B ∈ R^(d_H×r) - shared head dimension factors
    - C_h ∈ R^(H×r) - attention head factors
    - C_l ∈ R^(L×r) - layer factors
    - C_m ∈ R^(M×r) - matrix type factors
    """

    def __init__(self,
                 damping: float = 1e-4,
                 update_frequency: int = 1,
                 use_float16: bool = False,
                 diagnostic: bool = False):
        """
        Args:
            damping: Regularisation parameter ε for matrix inversion stability
            update_frequency: Apply preconditioning every N steps
            use_float16: Use half precision for memory efficiency
            diagnostic: Print diagnostic information
        """
        self.damping = damping
        self.update_frequency = update_frequency
        self.use_float16 = use_float16
        self.diagnostic = diagnostic
        self.step_count = 0

        # Cached preconditioner matrices
        self.preconditioners: Dict[str, torch.Tensor] = {}

    def _matrixinverse(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute regularised matrix inverse with numerical stability

        Args:
            matrix: Square matrix to invert

        Returns:
            Regularised inverse matrix
        """
        device = matrix.device
        dtype = matrix.dtype
        size = matrix.size(-1)

        # Add damping regularisation: M^(-1) = (M + εI)^(-1)
        regularised = matrix + self.damping * torch.eye(size, device=device, dtype=dtype)

        try:

            if torch.allclose(matrix, matrix.T, rtol=1e-6):
                try:
                    L = torch.linalg.cholesky(regularised)
                    inv = torch.cholesky_inverse(L)
                    return inv
                except RuntimeError:
                    pass

            # Standard matrix inverse
            inv = torch.linalg.inv(regularised)
            return inv

        except RuntimeError as e:
            if self.diagnostic:
                warnings.warn(f"Matrix inversion failed: {e}. Using pseudo-inverse.")
            # Fallback to pseudo-inverse for singular matrices
            return torch.linalg.pinv(regularised, rcond=1e-6)

    def compute_preconditioner_A(self,
                                A: torch.Tensor,
                                B: torch.Tensor,
                                C_h: torch.Tensor,
                                C_l: torch.Tensor,
                                C_m: torch.Tensor) -> torch.Tensor:

        #Compute preconditioner PA, this follows equation 6.1 in the urss paper:

       # P_A^(-1) = Σ_(k,ℓ,i) D_(k,ℓ,i) B^T B D_(k,ℓ,i)^T + εI

        #where D_{k,ℓ,i} = Diag(C_h[k,:]) Diag(C_l[ℓ,:]) Diag(C_m[i,:])

        #args:
           # A: shared embedding factor matrix (d×r)
            #B: shared head dimension factor matrix (d_H×r)
            #C_h: head factor matrix (H×r)
            #C_l: layer factor matrix (L×r)
            #C_m: matrix type factor matrix (M×r)

        #returns:
            #Preconditioner matrix P_A (r×r)

        H, L, M = C_h.size(0), C_l.size(0), C_m.size(0)  #here we are obtaining the row counts of the three matrices
        r = A.size(1)   #rank of A
        device = A.device  #here i am just checking whether A uses CPU or GPU
        dtype = torch.float16 if self.use_float16 else A.dtype  #choosing the precision

        # Initialise accumulator for Hessian approximation
        hessian_approx = torch.zeros(r, r, device=device, dtype=dtype)

        # here we are just expressing B^T B
        BtB = (B.T @ B).to(dtype)  # (r×r)



        #here we are iterating over attention heads, matrix types and layers
        # Sum over ALL!! (k,l,i) combinations
        #The shared factors A and B collect gradients from all (k,l,i) combinations
        combination_count = 0
        for k in range(H):
            for l in range(L):
                for i in range(M):
                    combination_count += 1

                    # Extract factor vectors
                    C_h_k = C_h[k, :].to(dtype)      # (r,) obtaining row k from head matrix
                    C_l_l = C_l[l, :].to(dtype)  #same dim. jhere
                    C_m_i = C_m[i, :].to(dtype)      # same dim. here

                    # Combined diagonal vector, D(k,ℓ,i)
                    d_combined = C_h_k * C_l_l * C_m_i  # (r,)

                    # Create diagonal matrix from combined factors
                    D_diag = torch.diag(d_combined)  # (r×r)

                    # Accumulate-D_(k,l,i) B^T B D_(k,l,i)^T
                    term = D_diag @ BtB @ D_diag.T
                    hessian_approx += term

       #here we are checking whether we actually summed over all possible combinations, returns an error if we didnt

        if self.diagnostic:
            print(f"P_A computation: summed over {combination_count} combinations (H={H}, L={L}, M={M})")
            print(f"Expected combinations: {H*L*M}")
            assert combination_count == H*L*M, "Combination count mismatch!"

        # Apply regularization and invert
        preconditioner = self._matrixinverse(hessian_approx)   #using the safe matrix inverse function defined previously
        return preconditioner.to(A.dtype)

    def compute_preconditioner_B(self,
                                A: torch.Tensor,
                                B: torch.Tensor,
                                C_h: torch.Tensor,
                                C_l: torch.Tensor,
                                C_m: torch.Tensor) -> torch.Tensor:
        """
        Compute preconditioner P_B following equation 6.2:



        P_B^(-1) = Σ_(k,ℓ,i) D_(k,ℓ,i)^T A^T A D_(k,ℓ,i) + εI

        Args:
            A: Shared embedding factor matrix (d×r)
            B: Shared head dimension factor matrix (d_H×r)
            C_h: Head factor matrix (H×r)
            C_l: Layer factor matrix (L×r)
            C_m: Matrix type factor matrix (M×r)

        Returns:
            Preconditioner matrix P_B (r×r)
        """

        #Essentially this is the same as for A, but we use the Gram matrix with respect to A this time to calucltate the preconditioner
        H, L, M = C_h.size(0), C_l.size(0), C_m.size(0)   #obtaining number of heads, rows, and matrices
        r = B.size(1)  #obtaining rank(number of columns)
        device = B.device
        dtype = torch.float16 if self.use_float16 else B.dtype

        # Initialise accumulator
        hessian_approx = torch.zeros(r, r, device=device, dtype=dtype)

        # Precompute A^T A
        AtA = (A.T @ A).to(dtype)  # (r×r) Gram matrix

        # Sum over ALL (k,ℓ,i) combinations

        combination_count = 0
        for k in range(H):
            for l in range(L):
                for i in range(M):
                    combination_count += 1

                    # Extract factor vectors
                    C_h_k = C_h[k, :].to(dtype)
                    C_l_l = C_l[l, :].to(dtype)
                    C_m_i = C_m[i, :].to(dtype)

                    # Combined diagonal vector
                    d_combined = C_h_k * C_l_l * C_m_i  # (r,)
                    D_diag = torch.diag(d_combined)  # (r×r)

                    # Accumulate: D_{k,l,i}^T A^T A D_{k,l,i}
                    term = D_diag.T @ AtA @ D_diag
                    hessian_approx += term


        #printing diagnostic information
        if self.diagnostic:
            print(f"P_B computation: summed over {combination_count} combinations")

        # Apply regularisation and invert
        preconditioner = self._matrixinverse(hessian_approx) #again use safe matrix inversse
        return preconditioner.to(B.dtype)

    def compute_preconditioner_CH(self,
                                 A: torch.Tensor,
                                 B: torch.Tensor,
                                 C_l: torch.Tensor,
                                 C_m: torch.Tensor) -> torch.Tensor:
        """
        Compute preconditioner P_CH following equation 6.3:

        P_CH^(-1) = Σ_(ℓ,i) (A^T A) ⊙ (D_L D_M B^T B D_M^T D_L^T) + εI

        where ⊙ denotes Hadamard product

        Args:
            A: Shared embedding factor matrix (d×r)
            B: Shared head dimension factor matrix (d_H×r)
            C_l: Layer factor matrix (L×r)
            C_m: Matrix type factor matrix (M×r)

        Returns:
            Preconditioner matrix P_CH (r×r)
        """
        L, M = C_l.size(0), C_m.size(0)  #obtaining number of layers and matrix types
        r = A.size(1)  #obtaining rank, number of columns
        device = A.device
        dtype = torch.float16 if self.use_float16 else A.dtype

        # Initialise accumulator, set to zeros
        hessian_approx = torch.zeros(r, r, device=device, dtype=dtype)

        # Precompute fixed terms
        AtA = (A.T @ A).to(dtype)  # (r×r) Gram matrix wrt A
        BtB = (B.T @ B).to(dtype)  # (r×r) Gram matrix wrt B

        # Sum over (ℓ,i) combinations
        combination_count = 0
        for l in range(L):
            for i in range(M):
                combination_count += 1

                # Extract factor vectors
                C_l_l = C_l[l, :].to(dtype)  # (r,)
                C_m_i = C_m[i, :].to(dtype)      # (r,)

                # Create diagonal matrices
                D_l = torch.diag(C_l_l)  # (r×r)
                D_i = torch.diag(C_m_i)    # (r×r)

                # Calculate D_L D_M B^T B D_M^T D_L^T
                BtB_term = D_l @ D_i @ BtB @ D_i.T @ D_l.T  # (r×r)

                # Hadamard product, (A^T A) ⊙ (D_L D_M B^T B D_M^T D_L^T)
                term = AtA * BtB_term
                hessian_approx += term

        if self.diagnostic:
            print(f"P_CH computation: summed over {combination_count} combinations (L={L}, M={M})")

        # Apply regularisation and invert
        preconditioner = self._matrixinverse(hessian_approx)   #use safe inverse matrix
        return preconditioner.to(A.dtype)

    def compute_preconditioner_CL(self,
                                 A: torch.Tensor,
                                 B: torch.Tensor,
                                 C_h: torch.Tensor,
                                 C_m: torch.Tensor) -> torch.Tensor:
        """
        Compute preconditioner P_CL following equation 6.4:

        P_CL^(-1) = Σ_{k,i} (D_H^T A^T A D_H) ⊙ (D_M B^T B D_M^T) + εI

        Args:
            A: Shared embedding factor matrix (d×r)
            B: Shared head dimension factor matrix (d_H×r)
            C_h: Head factor matrix (H×r)
            C_m: Matrix type factor matrix (M×r)

        Returns:
            Preconditioner matrix P_CL (r×r)
        """
        H, M = C_h.size(0), C_m.size(0)  #getting number of heads and number of matrix types
        r = A.size(1)  #rank, number of columns
        device = A.device
        dtype = torch.float16 if self.use_float16 else A.dtype

        # Initialise accumulator
        hessian_approx = torch.zeros(r, r, device=device, dtype=dtype)

        # Precompute fixed terms
        AtA = (A.T @ A).to(dtype)  # (r×r)
        BtB = (B.T @ B).to(dtype)  # (r×r)

        # Sum over (k,i) combinations
        combination_count = 0
        for k in range(H):
            for i in range(M):
                combination_count += 1

                # Extract factor vectors
                C_h_k = C_h[k, :].to(dtype)  # (r,)
                C_m_i = C_m[i, :].to(dtype)  # (r,)

                # Create diagonal matrices
                D_h = torch.diag(C_h_k)  # (r×r)
                D_i = torch.diag(C_m_i)  # (r×r)

                # Compute D_H^T A^T A D_H
                AtA_term = D_h.T @ AtA @ D_h  # (r×r)

                # Compute D_M B^T B D_M^T
                BtB_term = D_i @ BtB @ D_i.T  # (r×r)

                # Hadamard product
                term = AtA_term * BtB_term
                hessian_approx += term

        if self.diagnostic:
            print(f"P_CL computation: summed over {combination_count} combinations (H={H}, M={M})")

        # Apply regularization and invert
        preconditioner = self._matrixinverse(hessian_approx)
        return preconditioner.to(A.dtype)

    def compute_preconditioner_CM(self,
                                 A: torch.Tensor,
                                 B: torch.Tensor,
                                 C_h: torch.Tensor,
                                 C_l: torch.Tensor) -> torch.Tensor:
        """
        Compute preconditioner P_CM following equation 6.5:

        P_CM^(-1) = Σ_{k,ℓ} (D_L^T D_H^T A^T A D_H D_L) ⊙ (B^T B) + εI

        Args:
            A: Shared embedding factor matrix (d×r)
            B: Shared head dimension factor matrix (d_H×r)
            C_h: Head factor matrix (H×r)
            C_l: Layer factor matrix (L×r)

        Returns:
            Preconditioner matrix P_CM (r×r)
        """
        H, L = C_h.size(0), C_l.size(0)
        r = A.size(1)
        device = A.device
        dtype = torch.float16 if self.use_float16 else A.dtype

        # Initialise accumulator
        hessian_approx = torch.zeros(r, r, device=device, dtype=dtype)

        # Precompute fixed terms
        AtA = (A.T @ A).to(dtype)  # (r×r)
        BtB = (B.T @ B).to(dtype)  # (r×r)

        # Sum over (k,ℓ) combinations
        combination_count = 0
        for k in range(H):
            for l in range(L):
                combination_count += 1

                # Extract factor vectors
                C_h_k = C_h[k, :].to(dtype)      # (r,)
                C_l_l = C_l[l, :].to(dtype)  # (r,)

                # Create diagonal matrices
                D_h = torch.diag(C_h_k)    # (r×r)
                D_l = torch.diag(C_l_l)  # (r×r)

                # Compute D_L^T D_H^T A^T A D_H D_L
                AtA_term = D_l.T @ D_h.T @ AtA @ D_h @ D_l  # (r×r)

                # Hadamard product with B^T B
                term = AtA_term * BtB
                hessian_approx += term

        if self.diagnostic:
            print(f"P_CM computation: summed over {combination_count} combinations (H={H}, L={L})")

        # Apply regularization and invert
        preconditioner = self._matrixinverse(hessian_approx)
        return preconditioner.to(A.dtype)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    #this is a method for convenience
    def compute_all_preconditioners(self,
                                  A: torch.Tensor,
                                  B: torch.Tensor,
                                  C_h: torch.Tensor,
                                  C_l: torch.Tensor,
                                  C_m: torch.Tensor) -> Dict[str, torch.Tensor]:

        #master function, instead of computing each preconditioner individually-we compute them in one go.
        """
        Compute all preconditioner matrices following equations 6.1-6.5

        Args:
            A: Shared embedding factor matrix (d×r)
            B: Shared head dimension factor matrix (d_H×r)
            C_h: Head factor matrix (H×r)
            C_l: Layer factor matrix (L×r)
            C_m: Matrix type factor matrix (M×r)

        Returns:
            Dictionary mapping parameter names to preconditioner matrices
        """
        if self.diagnostic:
            print(f"Computing preconditioners for tensor dimensions:")
            print(f"  A: {A.shape}, B: {B.shape}")
            print(f"  C_h: {C_h.shape}, C_l: {C_l.shape}, C_m: {C_m.shape}")

        preconditioners = {
            'A': self.compute_preconditioner_A(A, B, C_h, C_l, C_m),
            'B': self.compute_preconditioner_B(A, B, C_h, C_l, C_m),
            'C_h': self.compute_preconditioner_CH(A, B, C_l, C_m),
            'C_l': self.compute_preconditioner_CL(A, B, C_h, C_m),
            'C_m': self.compute_preconditioner_CM(A, B, C_h, C_l)
        }

        # cache the preconditioners
        self.preconditioners = preconditioners

        if self.diagnostic:
            for name, P in preconditioners.items():
                print(f"  P_{name}: {P.shape}, condition number: {torch.linalg.cond(P).item():.2e}")

        return preconditioners
        #returns a dictionary mapping parameter names to their preconditioner matrices

    def apply_preconditioning(self,
                            A: torch.Tensor,
                            B: torch.Tensor,
                            C_h: torch.Tensor,
                            C_l: torch.Tensor,
                            C_m: torch.Tensor) -> None:
     #REMEMBER TO IMPLEMENT STEP COUNT

        """
        Apply preconditioning to parameter gradients following equations:

        A_t = A_{t-1} - η_A (∇A L) P_A
        B_t = B_{t-1} - η_B (∇B L) P_B
        C_h[k,:]_t = C_h[k,:]_{t-1} - η_Ch P_Ch (∇C_h[k,:] L)
        C_l[ℓ,:]_t = C_l[ℓ,:]_{t-1} - η_Cl P_Cl (∇C_l[ℓ,:] L)
        C_m[i,:]_t = C_m[i,:]_{t-1} - η_Cm P_Cm (∇C_m[i,:] L)

        Args:
            A: Shared embedding factor matrix with gradients
            B: Shared head dimension factor matrix with gradients
            C_h: Head factor matrix with gradients
            C_l: Layer factor matrix with gradients
            C_m: Matrix type factor matrix with gradients
        """
        self.step_count += 1

        # Only apply preconditioning at specified frequency
        #computing preconditoners can be expensive, might be worth exploring how only applying preconditioners at specified intervals perfroms ********
        if self.step_count % self.update_frequency != 0:
            return

        #these lines of code are just safety checks

        # Verify all parameters have gradients, before trying to precondition them
        params_with_grads = []
        for name, param in [('A', A), ('B', B), ('C_h', C_h), ('C_l', C_l), ('C_m', C_m)]:
            if param.grad is not None:
                params_with_grads.append(name)
            else:
                if self.diag:
                    print(f"Warning: Parameter {name} has no gradient")
        #this avoids unnecessary computation
        if len(params_with_grads) == 0:
            if self.diagnostic:
                print("No gradients found, skipping preconditioning")
            return

        # Numerical stability check, case we get unexpected values or infinite values
        for name, param in [('A', A), ('B', B), ('C_h', C_h), ('C_l', C_l), ('C_m', C_m)]:
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    if self.diagnostic:
                        print(f"Non-finite gradients in {name}, skipping preconditioning")
                    return

        try:
            # compute preconditioners, use master method defined previously
            P = self.compute_all_preconditioners(A, B, C_h, C_l, C_m)

            # apply preconditioning to A, grad_A @ P_A
            if A.grad is not None:
                # A.grad is (d×r), P_A is (r×r), we get (d×r)
                A.grad.data = A.grad.data @ P['A']

            # apply preconditioning to B, grad_B @ P_B
            if B.grad is not None:
                # B.grad is (d_H×r), P_B is (r×r), we get (d_H×r)
                B.grad.data = B.grad.data @ P['B']


                #here we have a dimension mismatch, thus we do the following:
                """ # Transpose to make multiplication valid
                     C_h.grad.T  (r × H) , now each column is a gradient row

                    #  Apply preconditioner to each column
                     P['C_h'] @ C_h.grad.T  (r×r) @ (r×H) = (r×H)

                    #  Transpose back to get rows as intended
                    (P['C_h'] @ C_h.grad.T).T  (H×r), so we go back to original shape
                """

            #Apply preconditioning to C_h, P_CH @ grad_CH^T then transpose back
            if C_h.grad is not None:
                #C_h.grad is (H×r), P_Ch is (r×r)
                #apply row-wise, each row gets preconditioned independently
                preconditioned_Ch = (P['C_h'] @ C_h.grad.T).T  # (H×r)
                C_h.grad.data = preconditioned_Ch

            # Apply preconditioning to C_l, P_CL @ grad_CL^T, then  transpose back
            if C_l.grad is not None:
                # C_l.grad is (L×r), P_CL is (r×r)
                preconditioned_Cl = (P['C_l'] @ C_l.grad.T).T  # (L×r)
                C_l.grad.data = preconditioned_Cl

            # Apply preconditioning to C_m, P_CM @ grad_CM^T, then transpose back
            if C_m.grad is not None:
                # C_m.grad is (M×r), P_Cm is (r×r)
                preconditioned_Cm = (P['C_m'] @ C_m.grad.T).T  # (M×r)
                C_m.grad.data = preconditioned_Cm

            if self.diagnostic:
                print(f"Applied preconditioning at step {self.step_count}")

        except Exception as e:
            if self.diagnostic:
                print(f"Preconditioning failed at step {self.step_count}: {e}")
                import traceback
                traceback.print_exc()


def integrate_lorta_preconditioner(model,
                                 damping: float = 1e-4,
                                 update_frequency: int = 1,
                                 use_float16: bool = False,
                                 diagnostic: bool = False) -> LoRTAPreconditioner:
    """
    Integrate LoRTA preconditioner with a LoRTAModel instance

    Args:
        model: LoRTAModel instance from the repository
        damping: Regularisation parameter for matrix inversion
        update_frequency: Apply preconditioning every N steps
        use_float16: Use half precision for memory efficiency
        diagnostic: Print diagnostic information

    Returns:
        LoRTAPreconditioner instance

    """
    # Verify model has required LoRTA parameters
    required_params = ['lora_A', 'lora_B', 'lora_C_h', 'lora_C_l', 'lora_C_m']
    missing_params = []

    for param_name in required_params:
        if not hasattr(model.model, param_name):
            missing_params.append(param_name)

    if missing_params:
        raise ValueError(f"Model missing required LoRTA parameters: {missing_params}")

    # Create preconditioner instance
    preconditioner = LoRTAPreconditioner(
        damping=damping,
        update_frequency=update_frequency,
        use_float16=use_float16,
        diagnostic=diagnostic
    )

    if diagnostic:
        A = model.model.lora_A     #nested strcuture including PEFT wrapper
        B = model.model.lora_B
        C_h = model.model.lora_C_h
        C_l = model.model.lora_C_l
        C_m = model.model.lora_C_m

        print(f"LoRTA Preconditioner Integration:")
        print(f"  Parameter shapes:")
        print(f"    A: {A.shape} (d={A.size(0)}, r={A.size(1)})")
        print(f"    B: {B.shape} (d_H={B.size(0)}, r={B.size(1)})")
        print(f"    C_h: {C_h.shape} (H={C_h.size(0)}, r={C_h.size(1)})")
        print(f"    C_l: {C_l.shape} (L={C_l.size(0)}, r={C_l.size(1)})")
        print(f"    C_m: {C_m.shape} (M={C_m.size(0)}, r={C_m.size(1)})")
        print(f"  Total combinations: {C_h.size(0) * C_l.size(0) * C_m.size(0)}")
        print(f"  Damping: {damping}")
        print(f"  Update frequency: {update_frequency}")

    return preconditioner


# Training loop integration utility
def create_preconditioned_training_step(model, preconditioner, optimizer):
    """
    Create a training step function with integrated preconditioning

    Args:
        model: LoRTAModel instance
        preconditioner: LoRTAPreconditioner instance
        optimiser: optimiser

    Returns:
        Training step function
    """
    def training_step(batch, criterion):
        """
        Execute one training step with preconditioning

        Args:
            batch: Training batch
            criterion: Loss function

        Returns:
            Loss value
        """
        optimizer.zero_grad()  #clear previous gradient computations

        # Forward pass
        outputs = model(**batch)
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        else:
            loss = criterion(outputs.logits, batch['labels'])

        # Backward pass
        loss.backward()

        # Apply preconditioning
        preconditioner.apply_preconditioning(
            model.model.lora_A,
            model.model.lora_B,
            model.model.lora_C_h,
            model.model.lora_C_l,
            model.model.lora_C_m
        )

        # Gradient clipping (recommended for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimiser step
        optimizer.step()

        return loss.item()

    return training_step

#not strictly necessary but will be implemented later for cleaner code
class PreconditionedLoRTAOptimizer:
    """
    Optimizer wrapper that automatically applies LoRTA preconditioning
    """

    def __init__(self,
                 model,
                 optimizer,
                 preconditioner: LoRTAPreconditioner):
        """
        Args:
            model: LoRTAModel instance
            optimizer: Base PyTorch optimizer
            preconditioner: LoRTAPreconditioner instance
        """
        self.model = model
        self.optimizer = optimizer
        self.preconditioner = preconditioner

        # Verify model compatibility
        required_params = ['lora_A', 'lora_B', 'lora_C_h', 'lora_C_l', 'lora_C_m']
        for param_name in required_params:
            if not hasattr(model.model, param_name):
                raise ValueError(f"Model missing required parameter: {param_name}")

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        """
        Perform optimization step with automatic preconditioning

        Args:
            closure: Optional closure for line search optimizers

        Returns:
            Optimizer step result
        """
        # Apply preconditioning before optimizer step
        self.preconditioner.apply_preconditioning(
            self.model.model.lora_A,
            self.model.model.lora_B,
            self.model.model.lora_C_h,
            self.model.model.lora_C_l,
            self.model.model.lora_C_m
        )

        # Execute base optimizer step
        return self.optimizer.step(closure)

    @property
    def param_groups(self):
        """Access optimizer parameter groups"""
        return self.optimizer.param_groups

    @property
    def state(self):
        """Access optimizer state"""
        return self.optimizer.state
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#will use this later for testing
# Integration with Hugging Face Trainer
class LoRTATrainerCallback:
    """
    Trainer callback for automatic LoRTA preconditioning integration
    """

    def __init__(self, preconditioner: LoRTAPreconditioner):
        self.preconditioner = preconditioner

    def on_step_end(self, args, state, control, model, **kwargs):
        """Apply preconditioning after gradient computation but before optimizer step"""
        if hasattr(model, 'model') and hasattr(model.model, 'lora_A'):
            self.preconditioner.apply_preconditioning(
                model.model.lora_A,
                model.model.lora_B,
                model.model.lora_C_h,
                model.model.lora_C_l,
                model.model.lora_C_m
            )


# Testing and validation utilities
def validate_preconditioner_implementation(model,
                                         preconditioner: LoRTAPreconditioner,
                                         test_input_shape: Tuple[int, int] = (4, 128),
                                         diagnostic: bool = True) -> bool:
    """
    Validate that the preconditioner implementation works correctly

    Args:
        model: LoRTAModel instance
        preconditioner: LoRTAPreconditioner instance
        test_input_shape: Shape for test input (batch_size, seq_len)
        diagnostic: Print validation results

    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Create test input
        batch_size, seq_len = test_input_shape
        vocab_size = model.model.config.vocab_size if hasattr(model.model, 'config') else 50257

        test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
        test_target = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Forward pass
        model.train()
        outputs = model(test_input, labels=test_target)
        loss = outputs.loss

        # Backward pass to generate gradients
        loss.backward()

        # Verify gradients exist
        params_with_grads = []
        for name, param in [('A', model.model.lora_A), ('B', model.model.lora_B),
                          ('C_h', model.model.lora_C_h), ('C_l', model.model.lora_C_l),
                          ('C_m', model.model.lora_C_m)]:
            if param.grad is not None:
                params_with_grads.append(name)
            else:
                if diagnostic:
                    print(f"Warning: No gradient for {name}")

        if len(params_with_grads) == 0:
            if diagnostic:
                print("Validation failed: No gradients found")
            return False

        # Store original gradients for comparison
        original_grads = {}
        for name, param in [('A', model.model.lora_A), ('B', model.model.lora_B),
                          ('C_h', model.model.lora_C_h), ('C_l', model.model.lora_C_l),
                          ('C_m', model.model.lora_C_m)]:
            if param.grad is not None:
                original_grads[name] = param.grad.clone()

        # Apply preconditioning
        preconditioner.apply_preconditioning(
            model.model.lora_A,
            model.model.lora_B,
            model.model.lora_C_h,
            model.model.lora_C_l,
            model.model.lora_C_m
        )

        # Verify gradients were modified
        modifications = {}
        for name, param in [('A', model.model.lora_A), ('B', model.model.lora_B),
                          ('C_h', model.model.lora_C_h), ('C_l', model.model.lora_C_l),
                          ('C_m', model.model.lora_C_m)]:
            if param.grad is not None and name in original_grads:
                diff = torch.norm(param.grad - original_grads[name]).item()
                modifications[name] = diff

                if diagnostic:
                    print(f"Parameter {name}: gradient modification magnitude = {diff:.2e}")

        # Check for numerical issues
        for name, param in [('A', model.model.lora_A), ('B', model.model.lora_B),
                          ('C_h', model.model.lora_C_h), ('C_l', model.model.lora_C_l),
                          ('C_m', model.model.lora_C_m)]:
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    if diagnostic:
                        print(f"Validation failed: Non-finite gradients in {name}")
                    return False

        # Verify preconditioners were computed
        if len(preconditioner.preconditioners) != 5:
            if diagnostic:
                print(f"Validation failed: Expected 5 preconditioners, got {len(preconditioner.preconditioners)}")
            return False

        # Check preconditioner properties
        for name, P in preconditioner.preconditioners.items():
            # Check symmetry (preconditioners should be symmetric)
            if not torch.allclose(P, P.T, rtol=1e-4):
                if diagnostic:
                    print(f"Warning: Preconditioner P_{name} is not symmetric")

            # Check positive definiteness (eigenvalues should be positive)
            try:
                eigenvals = torch.linalg.eigvals(P).real
                if (eigenvals <= 0).any():
                    if diagnostic:
                        print(f"Warning: Preconditioner P_{name} has non-positive eigenvalues")
            except RuntimeError:
                if diagnostic:
                    print(f"Warning: Could not compute eigenvalues for P_{name}")

        if diagnostic:
            print(f"Validation passed: Preconditioning applied successfully")
            print(f"Parameters with gradients: {list(modifications.keys())}")
            total_modification = sum(modifications.values())
            print(f"Total gradient modification magnitude: {total_modification:.2e}")

        return True

    except Exception as e:
        if diagnostic:
            print(f"Validation failed with exception: {e}")
            import traceback
            traceback.print_exc()
        return False





#ADD QUOTE MARKS HERE !!! to wrap writing to preconditioner file
