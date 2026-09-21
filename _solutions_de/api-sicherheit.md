---
title: API-Sicherheit
description: Absicherung von APIs durch Rate Limiting, Schema-Validierung, Gateways
  und tokenbasierte Authentifizierung.
category:
- Security
- Architecture
problems:
- rate-limiting-issues
- authentication-bypass-vulnerabilities
- authorization-flaws
- high-api-latency
- rest-api-design-issues
- sql-injection-vulnerabilities
- legacy-api-versioning-nightmare
- data-protection-risk
- cross-site-scripting-vulnerabilities
- graphql-complexity-issues
layout: solution
lang: de
en_slug: api-security
related_solutions:
- slug: authentication
  similarity: 0.75
- slug: api-first-design
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.7
- slug: data-flow-control
  similarity: 0.7
- slug: api-documentation
  similarity: 0.7
- slug: encryption
  similarity: 0.7
---

## Description

API-Sicherheit ist die Anwendung geschichteter Kontrollen — tokenbasierte Authentifizierung, Rate Limiting, Anfrage-Schema-Validierung, Antwortfilterung und Transportverschlüsselung — zum Schutz einer API vor Missbrauch und Ausnutzung, typischerweise durchgesetzt an einem zentralisierten Punkt wie einem API-Gateway statt verstreut im gesamten Anwendungscode. Viele Legacy-APIs waren ursprünglich für eine kleine Menge vertrauenswürdiger interner Konsumenten hinter einer Unternehmens-Firewall designt, unter Nutzung von Authentifizierungsmechanismen wie Basic Auth oder IP-Positivlisten, die nur so lange angemessenen Schutz boten, wie diese Perimeter-Annahme galt; während diese APIs schrittweise für Partner, mobile Clients und Drittintegrationen geöffnet werden, entspricht das ursprüngliche Bedrohungsmodell nicht mehr, wie die API tatsächlich genutzt und exponiert wird. Weil Legacy-Anwendungscode oft riskant oder langsam sicher zu ändern ist, ist der praktische Ausgangspunkt, diese Kontrollen an einem Gateway vor dem Legacy-Backend hinzuzufügen, was es erlaubt, Authentifizierung, Drosselung und Eingabevalidierung zu härten, ohne Quellcode zu berühren, den kaum noch jemand vollständig versteht. Schema-Validierung und Antwortfilterung auf dieser Ebene kompensieren außerdem Legacy-Code, der unerwartete oder fehlerhafte Eingaben möglicherweise nicht sicher behandelt, und fangen Angriffe wie Injection-Versuche ab, bevor sie überhaupt Backend-Logik erreichen, die nie mit feindseliger Eingabe im Sinn geschrieben wurde. Dieser Ansatz tauscht eine geringe zusätzliche Latenz und ein neues Stück kritischer Infrastruktur gegen eine substanzielle und schnelle Verringerung der Angriffsfläche — Schutzmaßnahmen, die sonst Monate sorgfältiger, riskanter Refaktorierung innerhalb der Legacy-Codebasis erfordern würden, können stattdessen am Gateway innerhalb von Tagen deployt werden.

## How to Apply ◆

> Legacy-APIs fehlt es häufig an grundlegenden Sicherheitskontrollen, da sie für internen Gebrauch hinter Firewalls designt wurden, die keinen angemessenen Schutz mehr bieten. API-Sicherheit härtet diese Schnittstellen gegen moderne Bedrohungen durch geschichtete Kontrollen.

- Deployen Sie ein API-Gateway vor Legacy-API-Endpunkten, um Authentifizierung, Rate Limiting und Anfragevalidierung zu zentralisieren. Das Gateway agiert als Sicherheitsschicht, die konfiguriert werden kann, ohne den Legacy-Anwendungscode zu ändern.
- Implementieren Sie tokenbasierte Authentifizierung (OAuth 2.0 oder API-Schlüssel mit HMAC-Signaturen), um jegliche Legacy-Authentifizierungsmechanismen wie Basic Auth über unverschlüsselte Verbindungen oder IP-basierte Zugriffskontrolle zu ersetzen.
- Fügen Sie Rate Limiting auf der API-Gateway-Ebene hinzu, um Missbrauch, Brute-Force-Angriffe und versehentliche Dienstverweigerung durch sich falsch verhaltende Clients zu verhindern. Konfigurieren Sie Limits pro Client und pro Endpunkt basierend auf erwarteten Nutzungsmustern.
- Implementieren Sie Anfrage-Schema-Validierung, um fehlerhafte oder unerwartete Eingaben abzulehnen, bevor sie das Legacy-Backend erreichen. Dies verhindert Injection-Angriffe und schützt vor unerwarteten Payloads, die Legacy-Code möglicherweise nicht sicher handhabt.
- Fügen Sie Antwortfilterung am Gateway hinzu, um Über-Exposition von Daten zu verhindern. Legacy-APIs geben häufig vollständige Datenbankeinträge zurück, einschließlich Felder, die der Client nicht benötigt und nicht sehen sollte (interne IDs, Audit-Felder, sensible Daten).
- Aktivieren Sie Mutual TLS (mTLS) für Service-zu-Service-API-Kommunikation, um sicherzustellen, dass beide Parteien authentifiziert sind und Traffic verschlüsselt ist, als Ersatz für Legacy-unverschlüsselte interne Kommunikationsmuster.
- Implementieren Sie API-Versionierung und Deprecation-Richtlinien, sodass unsichere Legacy-API-Versionen ausgemustert werden können, während Clients zu gesicherten Versionen migrieren.

## Tradeoffs ⇄

> API-Sicherheit bietet Defense-in-Depth für exponierte Schnittstellen, führt aber Latenz-Overhead und operative Komplexität ein, die verwaltet werden müssen.

**Vorteile:**

- Zentralisiert Sicherheitskontrollen am API-Gateway und ermöglicht den Schutz von Legacy-APIs, ohne ihren Quellcode zu ändern.
- Verhindert Missbrauch durch Rate Limiting und Drosselung und schützt Backend-Systeme davor, durch böswillige oder versehentliche Übernutzung überwältigt zu werden.
- Verringert die Angriffsfläche durch Validierung von Eingaben und Filterung von Ausgaben, bevor sie die Legacy-Anwendung erreichen oder verlassen.
- Ermöglicht schrittweise Sicherheitsverbesserung durch inkrementelles Hinzufügen von Kontrollen, ohne eine vollständige Neuschreibung von Legacy-API-Endpunkten zu erfordern.

**Kosten und Risiken:**

- Das API-Gateway fügt jeder Anfrage Latenz hinzu, was für latenzsensitive Legacy-Anwendungen spürbar sein kann.
- Tokenbasierte Authentifizierung erfordert, dass Clients aktualisiert werden, um den neuen Authentifizierungsablauf zu unterstützen, was für externe Konsumenten disruptiv sein kann.
- Übermäßig restriktive Rate Limits oder Schema-Validierungsregeln können legitime Nutzungsmuster brechen, besonders für Legacy-Clients mit nicht standardmäßigen Anfrageformaten.
- Das API-Gateway selbst wird zu einer kritischen Infrastrukturkomponente, die hochverfügbar und ordentlich gesichert sein muss.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie API-Sicherheit Legacy-Systeme vor modernen Bedrohungen schützt.

Ein Legacy-CRM-System exponiert eine REST-API, die ursprünglich für internen Gebrauch durch eine einzige Frontend-Anwendung designt war. Über die Jahre wurde die API mit Partnern, mobilen Apps und Drittintegrationen geteilt, alle unter Nutzung von Basic-Authentifizierung über HTTPS. Das System hat kein Rate Limiting, und ein falsch konfigurierter Batch-Job eines Partners sendet 50.000 Anfragen pro Minute, was die Legacy-Datenbank überlastet. Das Team deployt ein API-Gateway, das OAuth-2.0-Token-Authentifizierung durchsetzt, Rate Limits von 100 Anfragen pro Minute pro Client anwendet und Anfrage-Payloads gegen ein OpenAPI-Schema validiert. Das Gateway entfernt außerdem sensible Felder (interne Kunden-IDs, Audit-Zeitstempel) aus API-Antworten. Innerhalb eines Monats wird der unautorisierte Batch-Job blockiert, zwei Injection-Angriffsversuche werden durch Schema-Validierung abgefangen, und die Last des Legacy-Backends stabilisiert sich.

Eine Legacy-Zahlungsabwicklungs-API akzeptiert Transaktionsanfragen mit minimaler Eingabevalidierung und verlässt sich darauf, dass die aufrufende Anwendung wohlgeformte Daten sendet. Ein Sicherheitsaudit deckt auf, dass die API durch einen schlecht bereinigten Händler-ID-Parameter für SQL-Injection anfällig ist. Statt die Legacy-Codebasis zu ändern, konfiguriert das Team das API-Gateway so, dass alle eingehenden Parameter gegen strikte Muster validiert werden (Händler-IDs müssen einem UUID-Format entsprechen, Beträge müssen positive Zahlen sein, Währungscodes müssen aus einer erlaubten Liste stammen). Zusätzlich implementieren sie eine Web-Application-Firewall-Regel (WAF) am Gateway, die übliche SQL-Injection-Muster blockiert. Diese Gateway-seitigen Schutzmaßnahmen werden innerhalb einer Woche deployt, verglichen mit den geschätzten drei Monaten, die eine Refaktorierung der Legacy-Eingabebehandlung erfordert hätte.
