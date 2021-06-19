def rs2_deproject_pixel2point(self, pixel, init_depth_map):    
        '''
        Transformate image coordinate in pixel to world coordinate in meter
        input: 
            desire pixel, 
            init_depth_map(see "save_depth_info.py")
        output:
            world_coordinate[x, y, z] 
        '''
        if(type(self.cv_depth) is np.ndarray):
            pixel_x = pixel[0]
            pixel_y = pixel[1]
            init_depth_val = init_depth_map[pixel_y][pixel_x]

            intrin_fx  = 612.9651489257812
            intrin_fy  = 613.2086791992188

            intrin_ppx = 321.706
            intrin_ppy = 238.366
            coeffs = [0, 0, 0, 0, 0]
            
            tmp_x = (pixel_x - intrin_ppx)/intrin_fx
            tmp_y = (pixel_y - intrin_ppy)/intrin_fy

            r2 = (tmp_x*tmp_x) + (tmp_y*tmp_y)
            f = 1 + coeffs[0]*r2 + coeffs[1]*r2*r2 + coeffs[4]*r2*r2*r2
            tmp_x  = tmp_x*f 
            tmp_y  = tmp_y*f

            point_x = init_depth_val*tmp_x
            point_y = init_depth_val*tmp_y

            scale=1000
            
            world_point  = [point_x/scale, point_y/scale, init_depth_val/scale]

            # print("world_point_camera", world_point[0], world_point[1], world_point[2])
            
            ### ========== img coor (meter) to arm coor (meter ) ========== ###
            rs_pos = np.matrix([ [world_point[0]], [world_point[1]], [world_point[2]]])

            # position shift
            # pos_sft = np.matrix([ [0.03], [-0.02], [-0.29] ])
            pos_sft = np.matrix([ [-0.03], [-0.29], [-0.02]])
            rs_pos = (pos_sft + rs_pos )

            # orientation rotate
            # base2rs = self.get_tf_matrix(  50,    0,   0)
            rs2base = self.get_tf_matrix(  125,    0,   0)
            rs_pos = np.linalg.inv(rs2base)*rs_pos

            # position shift2  (extra pos shift)
            # pos_sft = np.matrix([ [0.1373], [0.01], [0.0] ]) 
            # pos_sft = np.matrix([ [0.13], [0.064], [0.0] ])  
            # pos_sft = np.matrix([ [0.007], [-0.067], [0.09] ]) 
            # pos_sft = np.matrix([ [-0.003], [-0.067], [0.09] ])  # z 0.13

            pos_sft = np.matrix([ [-0.010], [-0.050], [0.15] ])  # Adjust this one!!!!!!!!!
            rs_pos = (rs_pos + pos_sft )

            world_pos = rs_pos

            # convert to float list
            world_pos = [ float(world_pos[0]), float(world_pos[1]), float(world_pos[2]) ]


            ### calculate the angle between camera and obj ###
            # tmp_ang = radians(self.env_angle_map[pixel[1], pixel[0]])
            # d_rs    = self.cv_depth[pixel[1]][pixel[0]]/1000
            # world_pos[2] = cos(tmp_ang)*d_rs

            # z_shift = 0.15
            world_pos[0] =  round(world_pos[0], 4)
            world_pos[1] =  round(world_pos[1], 4)
            world_pos[2] =  round(world_pos[2], 4) 
            # world_pos[2] = world_point[2]
            # print("world_position", world_pos[0], world_pos[1], world_pos[2])

            return world_pos
        
        else:
            return
