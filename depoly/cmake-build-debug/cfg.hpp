#include "unitree/common/json/jsonize.hpp"
#include <vector>
#include <iostream>

namespace unitree::common
{
    class ExampleCfg : public Jsonize
    {
    public:
        ExampleCfg() : kp(0), kd(0), dt(0), action_scale(0), pos_scale(0), vel_scale(0),history_length(0)
        {
        }

        void fromJson(JsonMap &json)
        {
            FromJson(json["kp"], kp);
            FromJson(json["kd"], kd);
            FromJson(json["dt"], dt);
            FromJson(json["action_scale"],action_scale);
            FromJson(json["pos_scale"],pos_scale);
            FromJson(json["vel_scale"],vel_scale);
            FromJson(json["history_length"],history_length);

            FromAny<float>(json["init_pos"], init_pos);
            FromAny<float>(json["init_gym_pos"], init_gym_pos);
        }

        void toJson(JsonMap &json) const
        {
            ToJson(kp, json["kp"]);
            ToJson(kd, json["kd"]);
            ToJson(dt, json["dt"]);
            ToJson(action_scale,json["action_scale"]);
            ToJson(pos_scale, json["pos_scale"]);
            ToJson(vel_scale,json["vel_scale"]);
            ToJson(history_length,json["history_length"]);
       
            ToAny<float>(init_pos, json["init_pos"]);
            ToAny<float>(init_gym_pos, json["init_gym_pos"]);
        }

        float kp;
        float kd;
        float dt;
        float action_scale;
        float pos_scale;
        float vel_scale;
        int history_length;

        std::vector<float> init_pos;
        std::vector<float> init_gym_pos;
    };
}