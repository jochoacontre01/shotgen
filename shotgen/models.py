import numpy as np

class GeoModel:
    
    def __init__(self, nx, nz, v_base = 4500):
        self.nx = nx
        self.nz = nz
        self.X, self.Z = np.meshgrid(np.arange(self.nx), np.arange(self.nz), indexing='ij')
        self.vel = self.vel = v_base * np.ones((self.nx, self.nz))
        
    def _create_layer_interface(self, base_depth, amplitude, wavelength, phase=0, slope=0):
        """Helper to create wavy layer interfaces"""
        return (base_depth + 
                slope * (self.X - self.nx/2) + 
                amplitude * np.sin(2 * np.pi * self.X / wavelength + phase))
        
    def layered(self):
        """
        Generates a 'Compactive Layered' model.
        Features: Mostly horizontal stratigraphy with varying bed thicknesses (thick & thin)
        and normal faults to test frequency/resolution.
        """
        print("Generating Layered Resolution model...")
        
        # 1. Background Velocity (Compaction Gradient)
        # Create a gradient from 2000 m/s to 4500 m/s
        for i in range(self.nz):
            self.vel[:, i] = 2000 + (i / self.nz) * 2500

        # 2. Define Stratigraphy
        # We define layers by (Relative Top Depth, Thickness Fraction, Velocity)
        # We include some very thin beds for resolution testing
        beds = [
            (0.10, 0.15, 2100),  # Thick overburden
            (0.30, 0.01, 3500),  # THIN High-Vel Stringer (Resolution test)
            (0.35, 0.08, 2400),  # Medium Sand
            (0.44, 0.015, 1800), # Thin low-vel gas sand?
            (0.47, 0.015, 1800), # Another thin one (Interference pair)
            (0.55, 0.20, 3200),  # Massive Carbonate Platform
            (0.78, 0.005, 4500)  # Ultra-thin basement fracture
        ]
        
        # Apply horizontal layers (with slight dip/wobble for realism)
        wobble = 0.005 * self.nz * np.sin(2 * np.pi * self.X / self.nx)
        
        for top_pct, thick_pct, v in beds:
            z_top = (top_pct * self.nz) + wobble
            z_bot = ((top_pct + thick_pct) * self.nz) + wobble
            
            # Create mask for this layer
            mask = (self.Z >= z_top) & (self.Z < z_bot)
            self.vel[mask] = v

        # 3. Normal Fault 1 (Synthetic)
        print("Faulting...")
        f1_x = 0.35 * self.nx
        f1_angle = 1.6 # Steep normal fault
        f1_throw = int(0.06 * self.nz) # moderate offset
        
        # Iterate to shift columns
        for ix in range(self.nx):
            z_fault = int(f1_angle * (ix - f1_x))
            if 0 <= z_fault < self.nz:
                # Normal fault: Hanging wall (right side) moves DOWN
                # We shift the content below z_fault downwards
                col = self.vel[ix, :].copy()
                
                # Shift content down by 'throw'
                # The content originally at [z_fault : end] moves to [z_fault+throw : end]
                # The gap [z_fault : z_fault+throw] mimics the layer being dragged or filled
                
                shifted = np.roll(col[z_fault:], f1_throw)
                col[z_fault:] = shifted
                
                # Simple fill for the gap (repeat top pixel of the gap to simulate drag)
                # Or just let the roll wrap around (which puts bottom pixels at top - not physical)
                # Let's fill the gap with the velocity just above the fault to prevent "wrap-around" artifacts
                col[z_fault:z_fault+f1_throw] = col[z_fault-1] if z_fault > 0 else 1500
                
                self.vel[ix, :] = col

        # 4. Normal Fault 2 (Antithetic - dipping opposite way? Or just another one?)
        # Let's do a second synthetic fault further out
        f2_x = 0.75 * self.nx
        f2_angle = 1.3
        f2_throw = int(0.12 * self.nz) # Larger offset
        
        for ix in range(self.nx):
            # z = slope * (x - x0) + z_offset
            z_fault = int(f2_angle * (ix - f2_x) + 0.3*self.nz)
            if 0 <= z_fault < self.nz:
                col = self.vel[ix, :].copy()
                shifted = np.roll(col[z_fault:], f2_throw)
                col[z_fault:] = shifted
                col[z_fault:z_fault+f2_throw] = col[z_fault-1] if z_fault > 0 else 1500
                self.vel[ix, :] = col
    
        return self.vel
            
    def foothills(self):
        """
        Creates a 'Foothills' thrust belt model.
        Features: High-velocity sheets thrust over low-velocity sediments.
        """
        print("Generating Foothills Overthrust model...")

        # 1. Background Stratigraphy (Repeated sequence)
        # We create a simplified sequence of alternating Hard/Soft rock
        layer_thickness = 0.2 * self.nz
        
        # Create parallel dipping layers first
        dip_slope = 0.3 
        # Calculate 'stratigraphic depth' Z' = Z - slope*X
        strat_depth = self.Z - dip_slope * self.X
        
        # Map stratigraphic depth to velocities using modulo to repeat layers
        # Velocities: 2500 (Soft), 3500 (Hard), 4500 (Basement-like)
        cycle_len = layer_thickness * 3
        
        # Assign base velocities
        self.vel[:, :] = 3000 # Default
        
        # Vectorized layer assignment based on modular arithmetic
        # This creates infinite repeating dipping layers
        mod_depth = strat_depth % cycle_len
        mask_soft = mod_depth < layer_thickness
        mask_hard = (mod_depth >= layer_thickness) & (mod_depth < 2*layer_thickness)
        mask_base = mod_depth >= 2*layer_thickness
        
        self.vel[mask_soft] = 2400
        self.vel[mask_hard] = 3400
        self.vel[mask_base] = 4200

        # 2. Add a Major Thrust Fault
        # A thrust fault pushes deeper (older) rock up and over shallower rock.
        # We simulate this by taking a wedge of the model and shifting it upwards/left.
        
        fault_x_start = 0.2 * self.nx
        fault_angle = -0.5 # Steep dip, negative because Z increases down
        
        # Define Fault Plane: Z = m(X - x0) + z0
        fault_z_intercept = 0.8 * self.nz
        fault_plane = fault_angle * (self.X - fault_x_start) + fault_z_intercept
        
        # Mask for the "Hanging Wall" (The block that moves)
        # In a thrust, the block *above* the fault moves *up*
        hanging_wall_mask = self.Z < fault_plane
        
        # Shift amount (Thrust displacement)
        shift_z = int(-0.15 * self.nz) # Move UP
        shift_x = int(0.1 * self.nx)   # Move RIGHT (Overthrusting)
        
        # We need to extract the pixels from the Hanging Wall and move them
        # This is tricky with vectorization, so we'll use a simplified geometric approach:
        # 1. Save the original Hanging Wall velocities
        # 2. Fill the "hole" left behind (or ignore it since we are overwriting)
        # 3. Paste them into the new location
        
        # Simplification: Regenerate the layers with a Coordinate Offset for the Hanging Wall
        # New Strat Depth for Hanging Wall = (Z - shift_z) - slope*(X - shift_x)
        
        hw_strat_depth = (self.Z - shift_z) - dip_slope * (self.X - shift_x)
        hw_mod_depth = hw_strat_depth % cycle_len
        
        # Create temp velocity arrays for the Hanging Wall logic
        hw_vel = np.zeros_like(self.vel)
        hw_vel[hw_mod_depth < layer_thickness] = 2400
        hw_vel[(hw_mod_depth >= layer_thickness) & (hw_mod_depth < 2*layer_thickness)] = 3400
        hw_vel[hw_mod_depth >= 2*layer_thickness] = 4200
        
        # Overwrite the main velocity model WHERE we are in the Hanging Wall
        self.vel[hanging_wall_mask] = hw_vel[hanging_wall_mask]
        
        # 3. Add Topography / Weathering Layer
        topo_mask = self.Z < (0.3 * self.nz + 0.05 * self.nz * np.sin(self.X / (0.1*self.nx)))
        self.vel[topo_mask] = 340 # Air (0) or Weathering (1500) - let's use air for land
        self.vel[topo_mask] = 1500   # Let's say 0 is air, or 1500 if submerged. Let's assume land.
        
        # 4. Trap (Sub-thrust)
        # Often hydrocarbons are trapped BENEATH the thrust sheet in the footwall
        # Locate a spot just below the fault plane
        print("Adding sub-thrust trap...")
        trap_center_x = 0.5 * self.nx
        # Calculate fault depth at this X
        f_depth = fault_angle * (trap_center_x - fault_x_start) + fault_z_intercept
        
        trap_mask = (((self.X - trap_center_x)**2 / (0.05 * self.nx)**2) + 
                     ((self.Z - (f_depth + 0.1*self.nz))**2 / (0.03 * self.nz)**2)) < 1
        self.vel[trap_mask] = 2100 # Oil
        
        return self.vel

    def gas_chimney(self):
        """
        Creates a 'Gas Chimney' model.
        Features: Vertical chaotic zone with low velocity and sagging layers.
        """
        print("Generating Gas Chimney model...")

        # 1. Background: Gentle Anticline
        self.vel[:, :] = 2800
        
        # Create 5 layers
        for i in range(5):
            depth = (0.2 + i * 0.15) * self.nz
            # Add slight anticline curve
            curve = depth - 0.05 * self.nz * np.sin(np.pi * self.X / self.nx)
            self.vel[self.Z > curve] = 3000 + i * 200

        # 2. Water Layer
        self.vel[self.Z < 0.15 * self.nz] = 1500

        # 3. The Chimney
        chimney_center = 0.5 * self.nx
        chimney_width = 0.12 * self.nx
        
        # Mask for the vertical column
        # Add some random jitter to the edges so it's not a perfect box
        jitter = 0.02 * self.nx * np.random.randn(self.nx, self.nz)
        dist_from_center = np.abs(self.X - chimney_center) + jitter
        chimney_mask = (dist_from_center < (chimney_width / 2)) & (self.Z > 0.15 * self.nz)
        
        # 4. Apply Effects inside Chimney
        # Effect A: Velocity Push-down (Gas lowers velocity)
        # We take the existing velocity and multiply by 0.85
        self.vel[chimney_mask] *= 0.85
        
        # Effect B: Chaos / Noise
        # Add random noise to simulate scattering/fractures
        noise = np.random.normal(0, 150, (self.nx, self.nz))
        self.vel[chimney_mask] += noise[chimney_mask]
        
        # Effect C: Sagging (Geometric)
        # We physically shift pixels down inside the chimney to simulate collapse
        sag_amount = int(0.03 * self.nz)
        
        # We apply this by rolling columns in the center, weighted by distance to center
        # (Simplified: just shifting the whole chimney column down)
        chimney_indices = np.where(np.abs(np.arange(self.nx) - chimney_center) < chimney_width/2)[0]
        
        for ix in chimney_indices:
            # Shift column down
            col = self.vel[ix, :].copy()
            # Roll positive (down)
            col[int(0.15*self.nz):] = np.roll(col[int(0.15*self.nz):], sag_amount)
            self.vel[ix, :] = col
            
        # 5. Bright Spot (Gas accumulation at top of chimney)
        print("Adding shallow gas pocket...")
        top_chimney_z = 0.2 * self.nz
        gas_mask = (((self.X - chimney_center)**2 / (0.08 * self.nx)**2) + 
                    ((self.Z - top_chimney_z)**2 / (0.02 * self.nz)**2)) < 1
        self.vel[gas_mask] = 1600 # Very low velocity gas
        
        return self.vel

    def unconformity(self):
        """
        Creates an 'Angular Unconformity' model.
        Features: Steeply dipping layers truncated by a horizontal erosional surface.
        """
        print("Generating Angular Unconformity model...")

        # 1. Generate the "Old" Dipping Sequence (Bottom)
        dip_slope = 0.5
        layer_width = 0.1 * self.nz
        
        # Calculate projected depth Z_dip = Z - slope*X
        dip_z = self.Z - dip_slope * self.X
        
        # Modulo for repeating layers
        mod_dip = dip_z % (layer_width * 3)
        
        self.vel[:, :] = 3500 # Basement
        mask_1 = mod_dip < layer_width
        mask_2 = (mod_dip >= layer_width) & (mod_dip < 2*layer_width)
        
        self.vel[mask_1] = 2800 # Sandstone
        self.vel[mask_2] = 3200 # Shale
        # Remainder is Basement 3500
        
        # 2. Create the Unconformity Surface
        # A slightly wavy horizontal line
        unconformity_z = 0.4 * self.nz + 0.01 * self.nz * np.sin(self.X / (0.05*self.nx))
        
        # 3. Generate "New" Horizontal Sequence (Top)
        # Everything above unconformity_z is overwritten
        mask_above = self.Z < unconformity_z
        
        # Create standard horizontal layers
        # Layer 1
        self.vel[mask_above] = 2200
        
        # Layer 2 (Shallowest)
        mask_shallow = self.Z < (unconformity_z - 0.15 * self.nz)
        self.vel[mask_shallow] = 1800
        
        # Water
        self.vel[self.Z < 0.1 * self.nz] = 1500
        
        # 4. Channels (Incised into the top sequence)
        print("Cutting channels...")
        # Channel 1
        ch1_x = 0.3 * self.nx
        ch1_z = 0.3 * self.nz # Just above unconformity
        ch_radius = 0.04 * self.nx
        
        ch_mask = ((self.X - ch1_x)**2 + (self.Z - ch1_z)**2) < ch_radius**2
        # Cut half circle (only bottom half)
        ch_mask = ch_mask & (self.Z > ch1_z)
        self.vel[ch_mask] = 2500 # Sand filled channel
        
        # Channel 2
        ch2_x = 0.7 * self.nx
        ch2_z = 0.25 * self.nz 
        ch2_mask = ((self.X - ch2_x)**2 + (self.Z - ch2_z)**2) < ch_radius**2
        ch2_mask = ch2_mask & (self.Z > ch2_z)
        self.vel[ch2_mask] = 2600 # Sand filled
        
        # 5. Stratigraphic Trap (Truncation)
        # Oil trapped in a dipping layer just below the unconformity (sealed by shale above)
        print("Filling truncation trap...")
        
        # Find a location where a "Sandstone" dipping layer hits the unconformity
        # Visually approx based on slope 0.5
        trap_x = 0.6 * self.nx
        
        # Create a wedge shape below the unconformity
        trap_mask = (self.X > trap_x) & (self.X < trap_x + 0.05*self.nx) & \
                    (self.Z > unconformity_z) & (self.Z < unconformity_z + 0.05*self.nz)
        
        self.vel[trap_mask] = 2100 # Oil
        
        return self.vel
        
    def wrist(self):
        """
        Generates a biomedical cross-section of a human wrist (Transverse View).
        Uses typical ultrasound velocities for tissues.
        The outer skin layer adheres to the model edges, with no coupling fluid at the boundary.
        """
        print("Generating Wrist Phantom...")

        # --- Velocities (m/s) ---
        V_WATER = 1500     # Coupling fluid
        V_SKIN = 1450      # Skin/Fat (slower than muscle)
        V_MUSCLE = 1560    # Muscle/Soft Tissue
        V_BONE = 3200      # Cortical Bone (Radius/Ulna)
        V_MARROW = 1700    # Bone Marrow (inside bone)
        V_TENDON = 1650    # Tendons (Flexor)
        V_NERVE = 1540     # Median Nerve (similar to muscle, often slightly lower contrast)

        # 1. Background (Water Bath)
        self.vel[:, :] = V_WATER

        # Center coordinates
        cx = 0.5 * self.nx
        cz = 0.5 * self.nz

        # 2. Arm Outline (Skin/Fat Layer) - Elliptical Cross-section
        # Make the ellipse touch the model edges
        arm_width = self.nx
        arm_height = self.nz

        # Create elliptical mask for the arm
        arm_mask = (((self.X - cx)**2 / (arm_width/2)**2) +
                    ((self.Z - cz)**2 / (arm_height/2)**2)) < 1
        self.vel[arm_mask] = V_SKIN

        # Remove coupling fluid at the boundary: set all edge pixels that are skin to skin velocity
        # (already set above, but ensure no water at the edge where skin is present)
        # Optionally, you can enforce that all edge pixels are skin if they are inside the ellipse
        for edge in [0, self.nx-1]:
            self.vel[edge, :] = np.where(
                (((edge - cx)**2 / (arm_width/2)**2) + ((self.Z[edge, :] - cz)**2 / (arm_height/2)**2)) < 1,
                V_SKIN, self.vel[edge, :]
            )
        for edge in [0, self.nz-1]:
            self.vel[:, edge] = np.where(
                (((self.X[:, edge] - cx)**2 / (arm_width/2)**2) + ((edge - cz)**2 / (arm_height/2)**2)) < 1,
                V_SKIN, self.vel[:, edge]
            )

        # 3. Muscle Tissue (Inside skin, slightly smaller ellipse)
        muscle_width = arm_width - 0.05 * self.nx
        muscle_height = arm_height - 0.05 * self.nz
        muscle_mask = (((self.X - cx)**2 / (muscle_width/2)**2) +
                       ((self.Z - cz)**2 / (muscle_height/2)**2)) < 1
        self.vel[muscle_mask] = V_MUSCLE

        # 4. Bones (Radius and Ulna)
        # Radius (Larger, Lateral/Thumb side) - Let's put it on the Left
        radius_x = cx - 0.15 * self.nx
        radius_z = cz + 0.05 * self.nz
        radius_r = 0.24 * self.nx

        # Cortical Bone (Outer Shell)
        radius_mask = ((self.X - radius_x)**2 + (self.Z - radius_z)**2) < radius_r**2
        self.vel[radius_mask] = V_BONE
        # Marrow (Inner Core)
        radius_inner_mask = ((self.X - radius_x)**2 + (self.Z - radius_z)**2) < (radius_r * 0.7)**2
        self.vel[radius_inner_mask] = V_MARROW

        # Ulna (Smaller, Medial/Pinky side) - Right side
        ulna_x = cx + 0.30 * self.nx
        ulna_z = cz + 0.08 * self.nz # Ulna sits slightly posterior/dorsal usually
        ulna_r = 0.12 * self.nx

        ulna_mask = ((self.X - ulna_x)**2 + (self.Z - ulna_z)**2) < ulna_r**2
        self.vel[ulna_mask] = V_BONE
        ulna_inner_mask = ((self.X - ulna_x)**2 + (self.Z - ulna_z)**2) < (ulna_r * 0.6)**2
        self.vel[ulna_inner_mask] = V_MARROW

        # 5. Carpal Tunnel Region (Volar/Top side, between bones but superficial)
        # Median Nerve (Elliptical, usually slightly darker/slower)
        nerve_x = cx
        nerve_z = cz - 0.12 * self.nz # More superficial (closer to top)
        nerve_mask = (((self.X - nerve_x)**2 / (0.04 * self.nx)**2) +
                      ((self.Z - nerve_z)**2 / (0.02 * self.nz)**2)) < 1
        self.vel[nerve_mask] = V_NERVE

        # Flexor Tendons (High velocity spots around the nerve)
        # Let's add a cluster of 4 tendons
        tendon_radius = 0.015 * self.nx
        tendon_positions = [
            (nerve_x - 0.06*self.nx, nerve_z),
            (nerve_x + 0.06*self.nx, nerve_z),
            (nerve_x - 0.03*self.nx, nerve_z + 0.04*self.nz), # Deep flexors
            (nerve_x + 0.03*self.nx, nerve_z + 0.04*self.nz)
        ]

        for tx, tz in tendon_positions:
            t_mask = ((self.X - tx)**2 + (self.Z - tz)**2) < tendon_radius**2
            self.vel[t_mask] = V_TENDON

        return self.vel
    
    def diapir(self):
        """
        Generates a classic central salt diapir (mushroom shape) 
        with sediment layers dragged upwards against the salt body.
        """
        print("Generating central salt diapir model...")
        
        self.vel = 4500 * np.ones((self.nx, self.nz)) 
        
        # Parameters
        cx = 0.5 * self.nx  # Center X
        
        # 1. Define Seabed (Flat with slight noise)
        seabed_depth = 0.15 * self.nz
        
        # 2. Define Sedimentary Layers with "Drag" effect
        # We want layers to curve UPWARDS near the center (lower Z index)
        # We use an inverted Gaussian to pull layers up
        drag_width = 0.2 * self.nx
        drag_amplitude = -0.2 * self.nz  # Negative moves up in image coordinates
        
        drag_effect = drag_amplitude * np.exp(-((self.X - cx)**2) / (drag_width**2))
        
        # Layer definitions (Bottom up)
        # (Base Depth Fraction, Velocity)
        layers = [
            (0.85, 3400), # Deep sediments
            (0.65, 2900), # Mid-deep
            (0.45, 2400), # Reservoir level
            (0.30, 2000)  # Shallow
        ]
        
        # Reset background to basement
        self.vel[:, :] = 4000
        
        for base_frac, vel in layers:
            base_z = base_frac * self.nz
            # Layer interface = flat depth + drag effect
            interface = base_z + drag_effect
            
            # Clip interface so it doesn't breach the seabed
            interface = np.maximum(interface, seabed_depth + 10)
            
            mask = self.Z < interface
            self.vel[mask] = vel

        # 3. Apply Water Column (Overwrites sediments that dragged too high)
        self.vel[self.Z < seabed_depth] = 1500

        # 4. Create the Salt Diapir (Mushroom/Stock shape)
        # Stem: Vertical column
        stem_width = 0.08 * self.nx
        stem_top_z = 0.35 * self.nz
        
        stem_mask = (np.abs(self.X - cx) < stem_width) & (self.Z > stem_top_z)
        
        # Bulb/Cap: Elliptical shape on top
        bulb_center_z = 0.35 * self.nz
        bulb_width = 0.18 * self.nx
        bulb_height = 0.12 * self.nz
        
        bulb_mask = (((self.X - cx)**2 / bulb_width**2) + 
                     ((self.Z - bulb_center_z)**2 / bulb_height**2)) < 1
                     
        # Combine Stem and Bulb
        salt_mask = stem_mask | bulb_mask
        self.vel[salt_mask] = 4500  # Salt velocity

        # 5. Add Hydrocarbon Traps (Flank Traps)
        # Oil trapped against the salt stem, under the dragging layers
        print("Filling flank reservoirs...")
        
        # Left Flank Trap
        trap_z = 0.5 * self.nz
        trap_x_left = cx - stem_width - (0.02 * self.nx)
        
        # Create a small blob trapped against the salt
        trap_mask_left = (((self.X - trap_x_left)**2 / (0.04 * self.nx)**2) + 
                          ((self.Z - trap_z)**2 / (0.03 * self.nz)**2)) < 1
        
        # Ensure trap doesn't exist inside the salt
        trap_mask_left = trap_mask_left & ~salt_mask
        self.vel[trap_mask_left] = 2100 # Oil

        # Right Flank Trap
        trap_x_right = cx + stem_width + (0.02 * self.nx)
        trap_mask_right = (((self.X - trap_x_right)**2 / (0.04 * self.nx)**2) + 
                           ((self.Z - trap_z)**2 / (0.03 * self.nz)**2)) < 1
        trap_mask_right = trap_mask_right & ~salt_mask
        self.vel[trap_mask_right] = 2100 # Oil
        self._model_ready = True
        
        return self.vel
        
        
    def basin(self):
        """
        Generates a marine basin with water column, sediments, 
        salt structures, faults, and hydrocarbon traps.
        """
        print("Generating stratigraphy...")
        
        # --- 1. Define Stratigraphic Layers (Bottom up) ---
        # We use a list of (function_mask, velocity) tuples
        # Layers are defined by their bottom interface
        self.vel = 4500 * np.ones((self.nx, self.nz)) 
        
        # Deep Basement

        self.vel[:, :] = 4000 

        # Deep Sediments (slightly folded)
        interface_3 = self._create_layer_interface(base_depth=0.81 * self.nz, amplitude=0.05 * self.nz, wavelength=self.nx)
        mask = self.Z < interface_3
        self.vel[mask] = 3200

        # Middle Sediments (Anticline structure - potential trap)
        # We create a "hump" in the middle using a Gaussian-like shift on a sine wave
        interface_2 = (0.625 * self.nz) - (0.19 * self.nz) * np.exp(-((self.X - self.nx/2)**2) / ((0.375 * self.nx)**2))
        mask = self.Z < interface_2
        self.vel[mask] = 2800

        # Shallow Sediments
        interface_1 = (0.375 * self.nz) + (0.0125 * self.nz) * np.sin(self.X / (0.125 * self.nx))
        mask = self.Z < interface_1
        self.vel[mask] = 2200

        # --- 2. Marine Water Column ---
        # Water bottom (Seabed) - fairly flat with shelf slope
        seabed = (0.19 * self.nz) + 0.05 * self.X
        water_mask = self.Z < seabed
        self.vel[water_mask] = 1500

        # --- 3. Salt Diapir (Intrusion) ---
        # A large salt body rising from depth
        print("Injecting salt body...")
        salt_center_x = 0.25 * self.nx
        salt_center_z = 0.75 * self.nz
        # Elliptical equation for salt dome
        salt_mask = (((self.X - salt_center_x)**2 / (0.125 * self.nx)**2) + 
                     ((self.Z - salt_center_z)**2 / (0.31 * self.nz)**2)) < 1
        # Salt has high, constant velocity
        self.vel[salt_mask] = 4500 

        # --- 4. Faulting ---
        # We introduce a normal fault cutting through the anticline
        print("Faulting...")
        fault_x_start = 0.75 * self.nx
        fault_angle = 1.2 # slope
        
        # Calculate fault line Z = mx + c  => x = (z-c)/m
        # We want to shift everything to the right of the fault downwards (Normal Fault)
        fault_intercept = 0.375 * self.nz
        fault_plane_z = fault_angle * (self.X - fault_x_start) + fault_intercept
        
        # Apply shift: We roll the velocity array vertically in the faulted region
        # This is a simplification; accurate faulting requires interpolation, 
        # but 'roll' works for blocky models.
        throw = int(0.05 * self.nz) # pixels of vertical displacement
        
        # To do this cleanly with numpy, we need to iterate columns in the fault zone
        for ix in range(self.nx):
            z_fault = int(fault_angle * (ix - fault_x_start) + fault_intercept)
            if 0 <= z_fault < self.nz:
                # Everything below z_fault at this x gets shifted down
                col = self.vel[ix, :].copy()
                # Shift the sediment part down, filling top with overlay
                shifted_part = col[z_fault:-throw]
                col[z_fault+throw:] = shifted_part
                # (Optional) Fill the gap or let the layer above stretch
                # For simplicity here, we just apply the shift
                self.vel[ix, :] = col

        # --- 5. Hydrocarbons (Direct Hydrocarbon Indicators) ---
        print("Filling reservoirs...")
        
        # Trap 1: Anticline Crest Gas Cap (Bright spot / Low Velocity)
        # Located at the peak of the Middle Sediment layer (approx x=500, z=350)
        res_1_mask = (((self.X - 0.625 * self.nx)**2 / (0.075 * self.nx)**2) + 
                      ((self.Z - 0.45 * self.nz)**2 / (0.02 * self.nz)**2)) < 1
        # Gas significantly drops P-wave velocity
        self.vel[res_1_mask] = 1650 

        # Trap 2: Fault Trap (Oil)
        # Trapped against the fault plane we created earlier
        res_2_mask = (((self.X - 0.79 * self.nx)**2 / (0.05 * self.nx)**2) + 
                      ((self.Z - 0.56 * self.nz)**2 / (0.025 * self.nz)**2)) < 1
        # Cut off the reservoir at the fault line to make it look realistic
        fault_plane_z = fault_angle * (self.X - fault_x_start) + fault_intercept
        res_2_mask = res_2_mask & (self.Z < fault_plane_z) # Trapped *under* the seal
        self.vel[res_2_mask] = 2100
        self._model_ready = True
        
        return self.vel

    
    def layer_model(self):
        """
        Create a 4-layer horizontal velocity model.
        
        Assigns velocities of 1000, 1200, 800, and 1500 m/s to four 
        equal horizontal quadrants of the model.
        """
        vel = 1500 * np.ones((self.nx, self.nz))
        vel[:, int(self.nz/2):] = 3500
        
        self.vel = vel
        self._model_ready = True
        
        return self.vel
        
    def circle_model(self, radius=5):
        """
        Create a velocity model with a circular anomaly in the center.

        Parameters
        ----------
        radius : float, optional
            A divisor used to determine the circle radius (nz / radius). 
            Default is 5.
        """
        vel = 1000 * np.ones((self.nx, self.nz))
        # Center coordinates
        cx = self.nx // 2
        cz = self.nz // 2
        radius = self.nz / radius
        for ix in range(self.nx):
            for iz in range(self.nz):
                if ((ix - cx) ** 2 + (iz - cz) ** 2) < radius ** 2:
                    vel[ix, iz] = 3500
        self.vel = vel
        self._model_ready = True
        
        return self.vel