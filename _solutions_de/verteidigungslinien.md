---
title: Verteidigungslinien
description: Umsetzung von Sicherheitsmechanismen in mehreren Schichten und Ebenen.
category:
- Security
- Architecture
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- buffer-overflow-vulnerabilities
- data-protection-risk
- insecure-data-transmission
- cascade-failures
layout: solution
lang: de
en_slug: defense-lines
related_solutions:
- slug: network-segmentation
  similarity: 0.8
- slug: honeypots
  similarity: 0.75
- slug: authentication
  similarity: 0.75
- slug: logging-and-monitoring
  similarity: 0.75
- slug: encryption
  similarity: 0.75
- slug: authorization
  similarity: 0.7
---

## Description

Verteidigungslinien implementieren Sicherheit als eine Reihe unabhängiger Schichten — Netzwerksegmentierung, Transportverschlüsselung, Eingabevalidierung und Autorisierung auf Anwendungsebene, Zugriffskontrollen auf Datenbankebene, Verschlüsselung ruhender Daten und Überwachung —, sodass ein Ausfall oder Umgehen einer einzelnen Kontrolle nicht von sich aus zu einer vollständigen Kompromittierung des Systems führt. Der Mechanismus beruht auf dem Prinzip, dass kein kritischer Vermögenswert für seinen Schutz von genau einer Kontrolle abhängen sollte: Zum Beispiel kombiniert die Verteidigung gegen SQL-Injection Eingabevalidierung an der Anwendungsgrenze, parametrisierte Abfragen an der Datenzugriffsschicht und ein Least-Privilege-Datenbankkonto, das begrenzt, was ein Angreifer erreichen kann, selbst wenn die ersten beiden Schichten irgendwie besiegt werden. Dies ist für Legacy-Systeme wichtig, weil sie sich typischerweise mit einer einzigen dominanten Kontrolle entwickelt haben — einer Firewall, einer Login-Prüfung — implizit als ausreichend behandelt, eine in einer früheren Bedrohungsumgebung getroffene Annahme, die das System vollständig exponiert lässt, in dem Moment, in dem diese eine Kontrolle umgangen wird oder ihre zugrunde liegende Annahme (wie das Vertrauen in alles innerhalb des Netzwerkperimeters) nicht mehr gilt. Zusätzliche, unabhängige Kontrollen auf ein solches System zu schichten kann schrittweise erfolgen, wobei eine Verteidigungsschicht nach der anderen hinzugefügt wird, ohne ein vollständiges Sicherheits-Redesign zu erfordern, und jede zusätzliche Schicht erkauft auch Zeit für Erkennung und Reaktion, selbst in Fällen, in denen die davor liegenden präventiven Kontrollen schließlich versagen. Die entsprechenden Kosten sind architektonische und operative Komplexität: mehr Schichten bedeuten mehr Konfigurationsfläche zu verstehen und mehr sich Ende-zu-Ende ansammelnde Latenz, und überlappende Kontrollen können ein falsches Sicherheitsgefühl erzeugen, wenn Teams annehmen, dass eine andere Schicht eine Bedrohung handhabt, die in Wirklichkeit keine Schicht abdeckt.

## How to Apply ◆

> Legacy-Systeme verlassen sich typischerweise auf eine einzelne Sicherheitskontrolle (oft eine Firewall oder Login-Prüfung) als ihre einzige Verteidigung. Defense in Depth implementiert mehrere unabhängige Sicherheitsschichten, sodass ein Ausfall in einer einzelnen Kontrolle nicht zu einer vollständigen Kompromittierung führt.

- Bilden Sie die bestehenden Sicherheitskontrollen im Legacy-System ab und identifizieren Sie Schichten, in denen keine Kontrollen existieren. Häufige Lücken umfassen: Netzwerkschicht (keine Segmentierung), Transportschicht (keine Verschlüsselung), Anwendungsschicht (keine Eingabevalidierung), Datenschicht (keine Verschlüsselung im Ruhezustand) und Überwachungsschicht (keine Angriffserkennung).
- Implementieren Sie Sicherheitskontrollen auf jeder architektonischen Schicht unabhängig, sodass jede Schicht Schutz bietet, selbst wenn angrenzende Schichten kompromittiert sind: Perimeter-Firewalls, Netzwerksegmentierung, TLS-Verschlüsselung, Authentifizierung und Autorisierung auf Anwendungsebene, Eingabevalidierung, Zugriffskontrollen auf Datenbankebene und Datenverschlüsselung im Ruhezustand.
- Wenden Sie das Prinzip an, dass keine einzelne Sicherheitskontrolle die alleinige Verteidigung für einen kritischen Vermögenswert sein sollte. Zum Beispiel erfordert der Schutz vor SQL-Injection Eingabevalidierung an der Anwendungsgrenze, parametrisierte Abfragen an der Datenzugriffsschicht und Least-Privilege-Datenbankkonten, die Schaden begrenzen, selbst wenn Injection erfolgreich ist.
- Implementieren Sie Überwachung und Alarmierung als Verteidigungsschicht: Selbst wenn präventive Kontrollen versagen, können Erkennungskontrollen (Intrusion-Detection-Systeme, Anomalieerkennung, Audit-Logging) einen Verstoß identifizieren und dessen Auswirkung begrenzen.
- Fügen Sie Rate Limiting und Drosselung als Verteidigungsschicht hinzu, um automatisierte Angriffe zu verlangsamen und anderen Schichten Zeit zur Erkennung und Reaktion zu geben.
- Segmentieren Sie das Legacy-System in Sicherheitszonen mit unterschiedlichen Vertrauensstufen. Kommunikation zwischen Zonen sollte durch Sicherheits-Gateways laufen, die zonenspezifische Richtlinien durchsetzen.
- Implementieren Sie Fail-Secure-Standards: Wenn eine Sicherheitskontrolle ausfällt oder umgangen wird, sollte das System Zugriff verweigern, statt ihn zu erlauben. Legacy-Systeme scheitern oft offen und gewähren Zugriff, wenn Sicherheitsprüfungen nicht durchgeführt werden können.

## Tradeoffs ⇄

> Defense in Depth stellt sicher, dass kein einzelner Fehlerpunkt zu einem vollständigen Sicherheitsverstoß führt, erhöht aber die Systemkomplexität und kann die Performance beeinträchtigen.

**Vorteile:**

- Verhindert, dass eine einzelne Schwachstelle oder Fehlkonfiguration das gesamte System kompromittiert, indem sichergestellt wird, dass mehrere unabhängige Kontrollen besiegt werden müssen.
- Bietet Zeit für Erkennung und Reaktion, selbst wenn präventive Kontrollen versagen, was die Auswirkung erfolgreicher Angriffe begrenzt.
- Berücksichtigt die Realität, dass Legacy-Systeme immer einige Schwachstellen haben werden, indem sichergestellt wird, dass diese Schwachstellen nicht direkt ausnutzbar sind.
- Erlaubt schrittweise Sicherheitsverbesserung durch Hinzufügen von Verteidigungsschichten, ohne ein vollständiges Sicherheits-Redesign zu erfordern.

**Kosten und Risiken:**

- Mehrere Sicherheitsschichten fügen der Systemarchitektur Komplexität hinzu, was sie schwerer zu verstehen, zu konfigurieren und zu debuggen macht.
- Jede Sicherheitsschicht fügt Latenz und Rechenaufwand hinzu, die sich über mehrere Schichten hinweg summieren.
- Überlappende Kontrollen können ein falsches Sicherheitsgefühl erzeugen, wenn jede Schicht annimmt, dass eine andere Schicht eine bestimmte Bedrohung handhabt.
- Die Verwaltung und Pflege mehrerer Sicherheitsschichten erfordert mehr operativen Aufwand und Expertise als eine einzelne Kontrolle.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Defense in Depth vollständige Kompromittierung in Legacy-Systemen verhindert.

Eine Legacy-Webanwendung wird ausschließlich durch eine Web Application Firewall (WAF) geschützt, die bösartige Eingaben filtert. Als ein Angreifer eine WAF-Umgehungstechnik mittels Chunked Transfer Encoding entdeckt, injizieren sie erfolgreich SQL in die Anwendung und extrahieren die gesamte Kundendatenbank. Nach dem Vorfall implementiert das Team Defense in Depth: Die WAF bleibt als erste Schicht, wird aber durch parametrisierte Abfragen im Anwendungscode ergänzt (was SQL-Injection an der Quelle eliminiert), ein Least-Privilege-Datenbankkonto, das nur auf bestimmte Tabellen zugreifen kann, Spaltenverschlüsselung auf Datenbankebene für sensible Felder (sodass selbst extrahierte Daten verschlüsselt bleiben) und einen Datenbankaktivitätsmonitor, der bei ungewöhnlichen Abfragemustern alarmiert. Als ein nachfolgender Angreifer eine weitere WAF-Umgehung findet, blockieren die parametrisierten Abfragen die Injection. Das Team wird über die Überwachungsschicht über den Versuch informiert und patcht die WAF-Regel, was künftige Umgehungsversuche verhindert.

Ein Legacy-Finanzsystem verlässt sich auf Netzwerkperimeter-Sicherheit (Firewall) als primäre Verteidigung, unter der Annahme, dass alles innerhalb des Unternehmensnetzwerks vertrauenswürdig ist. Als der Laptop eines Mitarbeiters durch einen Phishing-Angriff kompromittiert wird, erhält der Angreifer uneingeschränkten Zugriff auf alle internen Systeme, einschließlich der administrativen Schnittstelle des Finanzsystems. Das Team implementiert Netzwerksegmentierung, um das Finanzsystem in seiner eigenen Sicherheitszone zu isolieren, fügt gegenseitiges TLS für alle Verbindungen zum Finanzsystem hinzu, verlangt Multi-Faktor-Authentifizierung für administrativen Zugriff, implementiert Autorisierung auf Anwendungsebene, die jede Anfrage gegen ein Berechtigungsmodell prüft, und setzt ein Intrusion-Detection-System ein, das auf anomale Zugriffsmuster überwacht. Als ein nachfolgender Phishing-Kompromiss auftritt, kann der Angreifer vom kompromittierten Arbeitsplatz aus nicht auf das Finanzsystem zugreifen, weil es in einem anderen Netzwerksegment liegt, und Versuche, das Segment des Finanzsystems zu erreichen, lösen einen Intrusion-Detection-Alarm aus.
