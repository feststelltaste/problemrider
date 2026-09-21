---
title: Autorisierung
description: Steuerung des Zugriffs auf Ressourcen basierend auf Berechtigungen.
category:
- Security
problems:
- authorization-flaws
- authentication-bypass-vulnerabilities
- data-protection-risk
- regulatory-compliance-drift
- password-security-weaknesses
- error-message-information-disclosure
layout: solution
lang: de
en_slug: authorization
related_solutions:
- slug: authorization-concept
  similarity: 0.85
- slug: authentication
  similarity: 0.8
- slug: domain-based-authorization-concept
  similarity: 0.8
- slug: role-based-access-control
  similarity: 0.8
- slug: least-privilege
  similarity: 0.75
- slug: data-flow-control
  similarity: 0.75
---

## Description

Autorisierung ist die Laufzeitdurchsetzung einer Zugriffskontrollentscheidung: Für jede Anfrage prüft sie, ob die authentifizierte Identität, die diese Anfrage stellt, die für die spezifische zugegriffene Ressource oder Operation erforderliche Berechtigung besitzt, und verweigert die Anfrage, wenn nicht. Sie unterscheidet sich von Authentifizierung, die nur feststellt, wer fragt, und sie unterscheidet sich von einem Autorisierungskonzept, welches das Design-Dokument ist statt des Mechanismus, der es ausführt. In Legacy-Systemen ist die Durchsetzung von Autorisierung oft verstreut — manche Endpunkte prüfen Berechtigungen, andere wurden später hinzugefügt und nie in die Prüfung eingebunden, und Batch-Jobs oder Berichtswerkzeuge umgehen die Anwendungsschicht vollständig und erreichen die Daten direkt —, was bedeutet, dass ein System an seinen Haupteinstiegspunkten sicher aussehen kann, während es durch Seiteneingänge weit offen bleibt. Diese Logik in einen einzigen Durchsetzungspunkt zu zentralisieren, konsistent sowohl auf API-Ebene als auch auf Datenzugriffsebene geprüft, und standardmäßig zu verweigern statt zu erlauben, schließt genau die Lücken, die sich anhäufen, wenn Autorisierungsprüfungen stückweise über die Lebenszeit eines Systems hinzugefügt werden, statt von Anfang an designt zu sein. Dies ist besonders wichtig in der Legacy-Modernisierung, weil jede neue Integration, jedes Migrationsskript oder jede Berichtserweiterung eine frische Gelegenheit ist, einen ungeprüften Pfad wieder einzuführen, es sei denn, die Durchsetzung ist zentralisiert statt ad hoc an jeder neuen Aufrufstelle wiederholt.

## How to Apply ◆

> Legacy-Systeme implementieren Autorisierung oft inkonsistent — manche Endpunkte prüfen Berechtigungen, während andere es nicht tun, oder Autorisierungslogik ist über die Codebasis ohne zentralen Durchsetzungspunkt verstreut. Systematische Autorisierung stellt sicher, dass jeder Zugriff auf Ressourcen und Operationen gegen ein definiertes Berechtigungsmodell verifiziert wird.

- Kartieren Sie alle geschützten Ressourcen und Operationen im Legacy-System und definieren Sie, wer Zugriff auf jede haben sollte. Viele Legacy-Systeme haben Ad-hoc-Zugriffsmuster angehäuft, bei denen Berechtigungen informell gewährt und nie überprüft wurden.
- Zentralisieren Sie Autorisierungslogik in einer einzigen Durchsetzungsschicht, statt Berechtigungsprüfungen über Controller, Services und Datenbankabfragen zu verstreuen. Ein zentralisierter Ansatz stellt Konsistenz sicher und macht es möglich, alle Zugriffskontrollentscheidungen zu auditieren.
- Implementieren Sie rollenbasierte Zugriffskontrolle (RBAC) als Baseline, wobei Nutzer auf Rollen und Rollen auf Berechtigungen abgebildet werden. Für komplexere Anforderungen erwägen Sie attributbasierte Zugriffskontrolle (ABAC), die Berechtigungen basierend auf Nutzerattributen, Ressourcenattributen und Umgebungsbedingungen bewertet.
- Fügen Sie Autorisierungsprüfungen sowohl auf der API-/Controller-Ebene als auch auf der Datenzugriffsebene hinzu. Prüfungen auf API-Ebene verhindern, dass unautorisierte Anfragen verarbeitet werden; Prüfungen auf Datenebene verhindern unautorisierten Zugriff über alternative Pfade (direkte Datenbankabfragen, Batch-Prozesse, Berichtswerkzeuge).
- Implementieren Sie das Deny-by-Default-Prinzip: Wenn keine explizite Berechtigung Zugriff gewährt, wird die Anfrage verweigert. Legacy-Systeme operieren oft nach einem impliziten Erlaubnis-Modell, bei dem neue Features für alle Nutzer zugänglich sind, es sei denn, jemand erinnert sich daran, Beschränkungen hinzuzufügen.
- Protokollieren Sie alle Autorisierungsentscheidungen (sowohl Gewährungen als auch Verweigerungen), um Audit-Anforderungen zu unterstützen und die Erkennung unautorisierter Zugriffsversuche zu ermöglichen.
- Führen Sie periodische Zugriffsüberprüfungen durch, um Berechtigungen zu entfernen, die nicht mehr benötigt werden, besonders für Nutzer, die ihre Rolle geändert haben oder die Organisation verlassen haben.

## Tradeoffs ⇄

> Ordentliche Autorisierung stellt sicher, dass Nutzer nur auf Ressourcen zugreifen und Operationen durchführen können, für die sie explizit berechtigt sind, erfordert aber umfassende Kartierung von Zugriffsanforderungen und konsistente Durchsetzung.

**Vorteile:**

- Verhindert unautorisierten Zugriff auf sensible Daten und Operationen, indem explizite Berechtigungsprüfungen auf jedem Zugriffspfad durchgesetzt werden.
- Unterstützt regulatorische Compliance-Anforderungen (GDPR, HIPAA, SOX), die Zugriffskontrolle und das Prinzip der geringsten Berechtigung vorschreiben.
- Bietet Audit-Fähigkeit, indem protokolliert wird, wer auf was zugegriffen hat und ob der Zugriff autorisiert war.
- Verringert den Explosionsradius kompromittierter Anmeldeinformationen — selbst wenn ein Angreifer gültige Anmeldeinformationen erhält, kann er nur auf Ressourcen zugreifen, die für die Rolle dieses Nutzers erlaubt sind.

**Kosten und Risiken:**

- Die Nachrüstung von Autorisierung in ein Legacy-System erfordert umfassende Identifikation aller Zugriffspfade, was für komplexe Systeme zeitaufwendig und fehleranfällig ist.
- Übermäßig restriktive Autorisierung kann bestehende Workflows brechen, die auf implizitem Zugriff beruhten, was Störungen während des Rollouts verursacht.
- Autorisierungslogik muss gepflegt werden, während sich das System weiterentwickelt; neue Features, die die Autorisierungsschicht umgehen, führen Schwachstellen wieder ein.
- Komplexe Berechtigungsmodelle (Hunderte von Rollen mit feingranularen Berechtigungen) werden schwierig zu verwalten und zu auditieren, was durch Fehlkonfiguration eigene Sicherheitsrisiken schafft.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Autorisierungskontrollen Zugriffsschwachstellen in Legacy-Systemen angehen.

Ein Legacy-Dokumentenmanagementsystem erlaubt jedem authentifizierten Nutzer den Zugriff auf jedes Dokument durch Ändern der Dokument-ID in der URL. Das System prüft, dass der Nutzer angemeldet ist, verifiziert aber nicht, dass er die Berechtigung hat, das angefragte Dokument anzusehen. Das Team implementiert ressourcenebenen Autorisierung, indem jedes Dokument mit einer Zugriffskontrollliste (ACL) verknüpft und eine Berechtigungsprüfung im Dokumentabrufservice hinzugefügt wird. Sie fügen außerdem eine zentralisierte Autorisierungs-Middleware hinzu, die alle API-Anfragen abfängt, die Ressourcenkennung extrahiert und verifiziert, dass die Rolle des authentifizierten Nutzers die erforderliche Berechtigung gewährt. Ein Sicherheitsscan nach dem Deployment bestätigt, dass Dokumentaufzählungsangriffe nicht mehr möglich sind — unautorisierte Anfragen erhalten eine 403-Forbidden-Antwort, ohne zu offenbaren, ob das Dokument existiert.

Ein Legacy-ERP-System hat über 15 Jahre 350 Nutzerrollen angehäuft, von denen viele überlappende oder übermäßige Berechtigungen gewähren. Mehrere für temporäre Projekte erstellte Rollen gewähren immer noch vollständigen administrativen Zugriff. Das Team führt eine Rollenkonsolidierung durch, indem tatsächliche Nutzungsmuster aus Zugriffsprotokollen kartiert werden, um zu identifizieren, welche Berechtigungen jeder Nutzer tatsächlich benötigt. Sie reduzieren die Rollenanzahl von 350 auf 45 gut definierte Rollen, implementieren Deny-by-Default-Autorisierung und entfernen administrative Berechtigungen von 12 Konten, die sie unnötigerweise hatten. Zugriffsüberprüfungen werden vierteljährlich geplant, um erneute Berechtigungsanhäufung zu verhindern.
