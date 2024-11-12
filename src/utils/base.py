from pydantic import BaseModel, Field

class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

        @staticmethod
        def json_schema_extra(schema: dict, _):
            props = {k: v for k, v in schema.get('properties', {}).items() if not v.get("hidden", False)}
            schema["properties"] = props