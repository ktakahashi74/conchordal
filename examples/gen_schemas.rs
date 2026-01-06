use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use conchordal::life::scenario::{LifeConfig, SpawnMethod, TimbreGenotype};
use schemars::{Schema, schema_for};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = Path::new("docs/schemas");
    fs::create_dir_all(out_dir)?;

    write_schema_markdown("life_config", &schema_for!(LifeConfig), out_dir)?;
    write_schema_markdown("spawn_method", &schema_for!(SpawnMethod), out_dir)?;
    write_schema_markdown("timbre_genotype", &schema_for!(TimbreGenotype), out_dir)?;

    Ok(())
}

fn write_schema_markdown(name: &str, root: &Schema, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let mut output = String::new();
    append_schema_section(&mut output, "Root", root);

    let defs = schema_definitions(root);

    for (def_name, schema) in defs.iter() {
        output.push('\n');
        output.push_str(&format!("## {def_name}\n\n"));
        append_schema_section(&mut output, def_name, schema);
    }

    let path = out_dir.join(format!("{name}.md"));
    fs::write(path, output)?;
    Ok(())
}

fn append_schema_section(output: &mut String, label: &str, schema: &Schema) {
    match schema.as_bool() {
        Some(_) => {
            output.push_str(&format!("{label}: any\n"));
        }
        None => {
            let Some(obj) = schema.as_object() else {
                output.push_str(&format!("{label}: any\n"));
                return;
            };
            if let Some(table) = schema_object_table(obj) {
                output.push_str(&table);
                return;
            }

            if append_variant_tables(output, obj) {
                return;
            }

            let summary = schema_object_summary(obj);
            output.push_str(&format!("{summary}\n"));
        }
    }
}

fn schema_object_table(obj: &Map<String, Value>) -> Option<String> {
    let properties = obj
        .get("properties")
        .and_then(Value::as_object)
        .filter(|props| !props.is_empty())?;

    let mut out = String::new();
    out.push_str("| Field | Type | Description |\n");
    out.push_str("| --- | --- | --- |\n");

    let mut keys: Vec<_> = properties.keys().collect();
    keys.sort();

    for key in keys {
        let schema = &properties[key];
        let type_name = schema_type_name(schema);
        let description = schema_description(schema);
        out.push_str(&format!("| {key} | {type_name} | {description} |\n"));
    }

    out.push('\n');
    Some(out)
}

fn append_variant_tables(output: &mut String, obj: &Map<String, Value>) -> bool {
    let variants = obj
        .get("oneOf")
        .and_then(Value::as_array)
        .filter(|list| !list.is_empty())
        .or_else(|| {
            obj.get("anyOf")
                .and_then(Value::as_array)
                .filter(|list| !list.is_empty())
        });
    let Some(variants) = variants else {
        return false;
    };

    let mut emitted = false;
    for (idx, schema) in variants.iter().enumerate() {
        let name = variant_name(schema).unwrap_or_else(|| format!("Variant {idx}"));
        output.push_str(&format!("### {name}\n\n"));
        if let Some(obj) = schema.as_object() {
            if let Some(table) = schema_object_table(obj) {
                output.push_str(&table);
                emitted = true;
                continue;
            }
        }

        let summary = schema_summary(schema);
        output.push_str(&format!("{summary}\n\n"));
        emitted = true;
    }

    emitted
}

fn schema_object_summary(obj: &Map<String, Value>) -> String {
    if let Some(values) = obj.get("enum").and_then(Value::as_array) {
        let items = values
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        if !items.is_empty() {
            return format!("Enum values: {items}");
        }
    }

    if let Some(reference) = obj.get("$ref").and_then(Value::as_str) {
        return format!("Reference: {reference}");
    }

    "Schema details are complex; see JSON schema.".to_string()
}

fn schema_summary(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => "Schema details are complex; see JSON schema.".to_string(),
        Value::Object(obj) => schema_object_summary(obj),
        _ => "Schema details are complex; see JSON schema.".to_string(),
    }
}

fn schema_type_name(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => "any".to_string(),
        Value::Object(obj) => {
            if let Some(reference) = obj.get("$ref").and_then(Value::as_str) {
                return reference_name(reference);
            }

            if let Some(instance_type) = obj.get("type") {
                return match instance_type {
                    Value::String(t) => instance_type_name(t, obj),
                    Value::Array(list) => {
                        let mut names = list
                            .iter()
                            .filter_map(Value::as_str)
                            .map(|t| instance_type_name(t, obj))
                            .collect::<Vec<_>>();
                        names.sort();
                        names.join(" or ")
                    }
                    _ => "object".to_string(),
                };
            }

            if obj.get("oneOf").is_some() || obj.get("anyOf").is_some() {
                return "enum/union".to_string();
            }

            "object".to_string()
        }
        _ => "any".to_string(),
    }
}

fn instance_type_name(instance_type: &str, obj: &Map<String, Value>) -> String {
    match instance_type {
        "array" => array_item_type(obj).unwrap_or_else(|| "array".to_string()),
        "boolean" => "bool".to_string(),
        "integer" => "integer".to_string(),
        "null" => "null".to_string(),
        "number" => "number".to_string(),
        "object" => "object".to_string(),
        "string" => "string".to_string(),
        _ => "object".to_string(),
    }
}

fn array_item_type(obj: &Map<String, Value>) -> Option<String> {
    let items = obj.get("items")?;
    let item_schema = match items {
        Value::Object(_) | Value::Bool(_) => items,
        Value::Array(list) => list.first()?,
        _ => return None,
    };
    Some(format!("{}[]", schema_type_name(item_schema)))
}

fn schema_description(schema: &Value) -> String {
    match schema {
        Value::Bool(_) => String::new(),
        Value::Object(obj) => obj
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn reference_name(reference: &str) -> String {
    reference
        .rsplit('/')
        .next()
        .map(|s| s.to_string())
        .unwrap_or_else(|| reference.to_string())
}

fn variant_name(schema: &Value) -> Option<String> {
    let obj = schema.as_object()?;
    let props = obj.get("properties")?.as_object()?;
    let prop = props
        .get("mode")
        .or_else(|| props.get("core"))
        .or_else(|| props.get("type"))?;
    let prop_obj = prop.as_object()?;
    let values = prop_obj.get("enum")?.as_array()?;
    values
        .first()
        .and_then(Value::as_str)
        .map(|v| v.to_string())
}

fn schema_definitions(root: &Schema) -> BTreeMap<String, Schema> {
    let Some(obj) = root.as_object() else {
        return BTreeMap::new();
    };
    let defs = obj
        .get("$defs")
        .or_else(|| obj.get("definitions"))
        .and_then(Value::as_object);
    let Some(defs) = defs else {
        return BTreeMap::new();
    };
    defs.iter()
        .filter_map(|(k, v)| {
            let schema = Schema::try_from(v.clone()).ok()?;
            Some((k.to_string(), schema))
        })
        .collect()
}
